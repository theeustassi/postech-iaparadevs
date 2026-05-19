"""
Módulo de segurança, auditoria e anonimização de dados sensíveis.

Responsabilidades:
- AuditLogger: registra todas as análises em arquivos JSONL diários
- Anonimizador: remove/mascara dados pessoais antes de qualquer persistência
- ValidadorEntrada: sanitiza parâmetros de entrada e detecta tentativas de injeção

Conformidade:
- LGPD (Lei 13.709/2018): dados de saúde são dados sensíveis e requerem
  tratamento especial, consentimento explícito e minimização de coleta.
- HIPAA (referência internacional): os princípios de mínima coleta,
  controle de acesso e registro de auditoria são seguidos.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Padroes de dados pessoais (Brasil)
_RE_CPF = re.compile(r"\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{2}")
_RE_TELEFONE = re.compile(r"(\(?\d{2}\)?\s?)?9?\d{4}[\s\-]?\d{4}")
_RE_DATA_NASCIMENTO = re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b")
_RE_NOME_COMPLETO = re.compile(r"([A-Z][a-záéíóúàèìòùãõâêîôûç]+\s){2,}[A-Z][a-záéíóúàèìòùãõâêîôûç]+")
_RE_CNS = re.compile(r"\b[1-9]\d{14}\b")

# Padroes de injecao (prompt injection, SQL injection, command injection)
_PADROES_INJECAO = [
    re.compile(r"ignore\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"(system|assistant|user)\s*:", re.IGNORECASE),
    re.compile(r"<\s*(script|iframe|object|embed)", re.IGNORECASE),
    re.compile(r";\s*(drop|delete|truncate|insert|update)\s+", re.IGNORECASE),
    re.compile(r"\|\s*(bash|sh|cmd|powershell)", re.IGNORECASE),
    re.compile(r"(rm|del)\s+-rf?", re.IGNORECASE),
]


class Anonimizador:
    """
    Remove ou substitui dados pessoais identificáveis (PII) de textos.

    Aplica substituições em cascata para CPF, CNS, telefone, datas e nomes.
    Os resultados anonimizados são seguros para log e relatório.
    """

    def anonimizar(self, texto: str) -> str:
        if not texto:
            return texto
        texto = _RE_CPF.sub("[CPF-ANONIMIZADO]", texto)
        texto = _RE_CNS.sub("[CNS-ANONIMIZADO]", texto)
        texto = _RE_TELEFONE.sub("[TEL-ANONIMIZADO]", texto)
        texto = _RE_DATA_NASCIMENTO.sub("[DATA-ANONIMIZADA]", texto)
        texto = _RE_NOME_COMPLETO.sub("[NOME-ANONIMIZADO]", texto)
        return texto

    def hash_id(self, identificador: str) -> str:
        """Retorna hash SHA-256 truncado (16 chars) de um identificador."""
        return hashlib.sha256(identificador.encode("utf-8")).hexdigest()[:16]

    def mascarar_id(self, identificador: str) -> str:
        """Preserva os primeiros 2 e últimos 2 caracteres, mascara o meio."""
        if len(identificador) <= 4:
            return "****"
        return identificador[:2] + "*" * (len(identificador) - 4) + identificador[-2:]


class ValidadorEntrada:
    """
    Valida e sanitiza entradas de texto do sistema.

    Detecta tentativas de prompt injection, SQL injection e command injection.
    Limita o tamanho das entradas para prevenir ataques de negação de serviço.
    """

    MAX_COMPRIMENTO_TEXTO = 10_000
    MAX_COMPRIMENTO_ID = 64

    def validar_id_paciente(self, paciente_id: str) -> bool:
        if not paciente_id or len(paciente_id) > self.MAX_COMPRIMENTO_ID:
            return False
        # Permite apenas alfanumericos, hifen e underscore
        return bool(re.match(r"^[\w\-]+$", paciente_id))

    def validar_texto(self, texto: str) -> tuple[bool, Optional[str]]:
        """
        Retorna (válido, motivo_de_rejeição).
        válido=True significa que o texto pode ser processado com segurança.
        """
        if len(texto) > self.MAX_COMPRIMENTO_TEXTO:
            return False, f"Texto muito longo ({len(texto)} caracteres, máximo {self.MAX_COMPRIMENTO_TEXTO})"

        for padrao in _PADROES_INJECAO:
            if padrao.search(texto):
                return False, f"Possível tentativa de injeção detectada no texto."

        return True, None

    def sanitizar(self, texto: str) -> str:
        """Remove caracteres de controle do texto."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)


class AuditLogger:
    """
    Logger estruturado para rastreabilidade de todas as análises do sistema.

    Cada evento é gravado como uma linha JSON em um arquivo diário (.jsonl),
    permitindo auditoria completa em conformidade com a LGPD. Os IDs de
    pacientes são mascarados antes de qualquer escrita.
    """

    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._anonimizador = Anonimizador()

    def _arquivo_do_dia(self) -> Path:
        data = datetime.utcnow().strftime("%Y%m%d")
        return self.log_dir / f"audit_{data}.jsonl"

    def _escrever(self, registro: Dict[str, Any]) -> None:
        with open(self._arquivo_do_dia(), "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    def registrar_analise_video(
        self,
        paciente_id: str,
        tipo_video: str,
        pontuacao_risco: float,
        n_alertas: int,
    ) -> None:
        self._escrever({
            "evento": "analise_video",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paciente_id_mascarado": self._anonimizador.mascarar_id(paciente_id),
            "tipo_video": tipo_video,
            "pontuacao_risco": round(pontuacao_risco, 4),
            "n_alertas": n_alertas,
        })
        logger.debug("Análise de vídeo registrada | Paciente: %s", self._anonimizador.mascarar_id(paciente_id))

    def registrar_analise_audio(
        self,
        paciente_id: str,
        tipo_consulta: str,
        pontuacao_risco: float,
        n_alertas: int,
    ) -> None:
        self._escrever({
            "evento": "analise_audio",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paciente_id_mascarado": self._anonimizador.mascarar_id(paciente_id),
            "tipo_consulta": tipo_consulta,
            "pontuacao_risco": round(pontuacao_risco, 4),
            "n_alertas": n_alertas,
        })

    def registrar_fusao(
        self,
        paciente_id: str,
        tipo_atendimento: str,
        nivel_prioridade: str,
        pontuacao_fusao: float,
    ) -> None:
        self._escrever({
            "evento": "fusao_multimodal",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paciente_id_mascarado": self._anonimizador.mascarar_id(paciente_id),
            "tipo_atendimento": tipo_atendimento,
            "nivel_prioridade": nivel_prioridade,
            "pontuacao_fusao": round(pontuacao_fusao, 4),
        })
        logger.info(
            "Fusão concluída | Paciente: %s | Prioridade: %s | Score: %.2f",
            self._anonimizador.mascarar_id(paciente_id),
            nivel_prioridade,
            pontuacao_fusao,
        )

    def registrar_alerta_critico(
        self,
        paciente_id: str,
        descricao: str,
    ) -> None:
        registro = {
            "evento": "alerta_critico",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paciente_id_mascarado": self._anonimizador.mascarar_id(paciente_id),
            "descricao_hash": hashlib.sha256(descricao.encode()).hexdigest()[:16],
        }
        self._escrever(registro)
        logger.warning(
            "ALERTA CRÍTICO | Paciente: %s",
            self._anonimizador.mascarar_id(paciente_id),
        )
