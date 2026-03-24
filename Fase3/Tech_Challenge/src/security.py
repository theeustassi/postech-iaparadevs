"""
Modulo de seguranca, logging de auditoria e validacao de entradas.

Responsabilidades:
- AuditLogger: registra todas as consultas e alertas em arquivos JSONL
- InputValidator: valida e sanitiza entradas do usuario
                  detecta tentativas de prompt injection

Em ambiente de producao, o AuditLogger deveria gravar em um banco de dados
imutavel (ex: banco de auditoria separado ou servico de logs centralizado).
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Logger estruturado para auditoria de todas as acoes do assistente medico.

    Cada evento e gravado como uma linha JSON em um arquivo diario (.jsonl),
    permitindo rastreabilidade completa e conformidade com requisitos regulatorios.

    Os IDs de pacientes sao parcialmente mascarados antes de serem gravados,
    mantendo rastreabilidade sem expor o identificador completo.
    """

    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_console_logger()

    def _setup_console_logger(self) -> None:
        """Configura o handler de console para o logger do modulo."""
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    def _current_log_file(self) -> Path:
        """Retorna o caminho do arquivo de log do dia atual."""
        date_str = datetime.utcnow().strftime("%Y%m%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _write(self, record: Dict[str, Any]) -> None:
        """Escreve um registro de auditoria no arquivo diario."""
        with open(self._current_log_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _mask_id(self, patient_id: str) -> str:
        """
        Mascara o ID do paciente para os logs.
        Mantém os primeiros 2 e ultimos 2 caracteres.
        """
        if len(patient_id) <= 4:
            return "****"
        return patient_id[:2] + ("*" * (len(patient_id) - 4)) + patient_id[-2:]

    def log_query(
        self,
        patient_id: str,
        query: str,
        response: str,
        severity: str,
        sources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registra uma consulta ao assistente com sua resposta.

        O conteudo da pergunta e da resposta nunca e armazenado diretamente;
        apenas seus hashes sao gravados, preservando a privacidade.
        """
        record = {
            "event": "query",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "patient_id_masked": self._mask_id(patient_id),
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "response_length_chars": len(response),
            "severity": severity,
            "sources": sources or [],
            "metadata": metadata or {},
        }
        self._write(record)
        logger.debug(
            f"Consulta registrada | Paciente: {self._mask_id(patient_id)} | "
            f"Gravidade: {severity}"
        )

    def log_alert(
        self,
        patient_id: str,
        alert_type: str,
        details: str,
    ) -> None:
        """Registra um alerta critico emitido pelo sistema."""
        record = {
            "event": "alert",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "patient_id_masked": self._mask_id(patient_id),
            "alert_type": alert_type,
            "details_hash": hashlib.sha256(details.encode()).hexdigest()[:16],
        }
        self._write(record)
        logger.warning(
            f"ALERTA [{alert_type}] emitido para paciente {self._mask_id(patient_id)}"
        )

    def log_safety_block(
        self,
        patient_id: str,
        blocked_content: str,
    ) -> None:
        """Registra um bloqueio de seguranca (ex: prescricao direta detectada)."""
        record = {
            "event": "safety_block",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "patient_id_masked": self._mask_id(patient_id),
            "blocked_content_hash": hashlib.sha256(
                blocked_content.encode()
            ).hexdigest()[:16],
        }
        self._write(record)
        logger.warning(
            f"Conteudo bloqueado por seguranca para paciente {self._mask_id(patient_id)}"
        )

    def log_error(
        self,
        patient_id: str,
        error_message: str,
        step: str = "desconhecido",
    ) -> None:
        """Registra erros do sistema para diagnostico e monitoramento."""
        record = {
            "event": "system_error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "patient_id_masked": self._mask_id(patient_id),
            "step": step,
            "error_hash": hashlib.sha256(error_message.encode()).hexdigest()[:16],
        }
        self._write(record)
        logger.error(
            f"Erro no passo '{step}' para paciente {self._mask_id(patient_id)}: "
            f"{error_message}"
        )

    def get_today_summary(self) -> Dict[str, int]:
        """Retorna um resumo dos eventos registrados hoje."""
        log_file = self._current_log_file()
        if not log_file.exists():
            return {"total": 0}

        counts: Dict[str, int] = {"total": 0}
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    event = rec.get("event", "unknown")
                    counts[event] = counts.get(event, 0) + 1
                    counts["total"] += 1
                except json.JSONDecodeError:
                    pass

        return counts


class InputValidator:
    """
    Valida e sanitiza inputs do usuario antes de serem processados pelo LLM.

    Detecta tentativas de prompt injection — uma classe de ataque onde o
    usuario tenta manipular o comportamento do LLM inserindo instrucoes
    no campo de entrada.
    """

    MAX_QUERY_LENGTH = 2000
    MAX_PATIENT_NOTES_LENGTH = 5000

    # Padroes associados a tentativas de prompt injection
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"ignore (all )?prior instructions",
        r"you are now",
        r"forget (all |your )?previous",
        r"disregard (all )?",
        r"system prompt",
        r"jailbreak",
        r"act as (if )?you (are|were)",
        r"pretend (you are|to be)",
        r"override your",
        r"bypass your",
    ]

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Valida a pergunta do usuario.

        Returns:
            tupla (valido: bool, mensagem_de_erro: str)
        """
        if not query or not query.strip():
            return False, "A pergunta nao pode estar vazia."

        if len(query) > self.MAX_QUERY_LENGTH:
            return (
                False,
                f"A pergunta excede o limite de {self.MAX_QUERY_LENGTH} caracteres.",
            )

        query_lower = query.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(
                    f"Tentativa de prompt injection detectada. Padrao: '{pattern}'"
                )
                return False, "Entrada invalida. Reformule sua pergunta."

        return True, ""

    def sanitize_text(self, text: str) -> str:
        """
        Remove caracteres de controle do texto, preservando newlines e tabs.
        """
        # Remove caracteres de controle exceto \n (0x0A) e \t (0x09)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()

    def validate_patient_info(
        self, patient_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Valida o dicionario de informacoes do paciente.

        Returns:
            tupla (valido: bool, mensagem_de_erro: str)
        """
        if "chief_complaint" not in patient_info:
            return False, "Campo obrigatorio ausente: 'chief_complaint' (queixa principal)."

        complaint = str(patient_info.get("chief_complaint", ""))
        if not complaint.strip():
            return False, "A queixa principal (chief_complaint) nao pode estar vazia."

        age = patient_info.get("age")
        if age is not None:
            try:
                age_val = float(age)
                if not (0 <= age_val <= 150):
                    return False, "Idade invalida. Deve estar entre 0 e 150."
            except (TypeError, ValueError):
                return False, "Idade deve ser um numero."

        return True, ""
