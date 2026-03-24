"""
Modulo de preprocessamento de dados medicos.

Responsabilidades:
- Carregar datasets no formato JSONL e JSON
- Anonimizar informacoes sensíveis (CPF, nomes, datas)
- Formatar os dados para fine-tuning no padrao instruction-following
- Dividir o dataset em treino, validacao e teste
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MedicalDataPreprocessor:
    """
    Preprocessa e anonimiza dados medicos para uso no pipeline de fine-tuning.

    O objetivo e garantir que nenhum dado de identificacao pessoal seja usado
    no treinamento do modelo, respeitando a LGPD e as boas praticas de IA medica.
    """

    # Padroes de dados sensiveis que devem ser removidos antes do treinamento
    SENSITIVE_PATTERNS = [
        (r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF_REMOVIDO]"),
        (r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", "[CNPJ_REMOVIDO]"),
        (r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b", "[NOME_REMOVIDO]"),
        (r"\b\d{2}/\d{2}/\d{4}\b", "[DATA_REMOVIDA]"),
        (r"\bCNS:\s*\d{15}\b", "[CNS_REMOVIDO]"),
        (r"\bRG:\s*[\d\.\-]+\b", "[RG_REMOVIDO]"),
        (r"\b\(\d{2}\)\s*\d{4,5}-\d{4}\b", "[TELEFONE_REMOVIDO]"),
    ]

    def anonymize_text(self, text: str) -> str:
        """
        Remove informacoes de identificacao pessoal do texto.

        Args:
            text: texto original que pode conter dados sensiveis

        Returns:
            texto com os dados sensiveis substituidos por marcadores
        """
        result = text
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result

    def hash_patient_id(self, patient_id: str) -> str:
        """
        Substitui o ID real do paciente por um hash SHA-256 truncado.
        Isso preserva a unicidade sem revelar o identificador original.
        """
        return "PAC-" + hashlib.sha256(patient_id.encode()).hexdigest()[:10]

    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Carrega um arquivo JSONL linha a linha."""
        records = []
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Linha {lineno} invalida em {filepath}: {e}")

        logger.info(f"Carregados {len(records)} registros de {filepath}")
        return records

    def load_patient_records(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Carrega registros de pacientes de um arquivo JSON e aplica anonimizacao.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)

        anonymized = []
        for rec in records:
            anon = dict(rec)
            if "patient_id" in anon:
                anon["patient_id"] = self.hash_patient_id(str(anon["patient_id"]))
            if "name" in anon:
                anon["name"] = "[ANONIMIZADO]"
            # Remove campos de identificacao direta
            for field in ("cpf", "rg", "cns", "phone", "email", "address"):
                anon.pop(field, None)
            if "notes" in anon:
                anon["notes"] = self.anonymize_text(anon["notes"])
            anonymized.append(anon)

        logger.info(f"Carregados e anonimizados {len(anonymized)} registros de pacientes")
        return anonymized

    def format_for_finetuning(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Converte os registros no formato padrao de instruction-following
        usado pelo SFTTrainer da biblioteca TRL.

        Formato de saida de cada exemplo:
            ### Instrucao: <instrucao do sistema>
            ### Contexto: <contexto clinico>
            ### Pergunta: <pergunta medica>
            ### Resposta: <resposta esperada>

        Args:
            records: lista de dicionarios com campos 'question', 'context' e 'answer'

        Returns:
            lista de dicionarios com campo 'text' pronto para o treinamento
        """
        system_instruction = (
            "Voce e um assistente medico especializado treinado com protocolos internos "
            "do HospitalIQ. Responda de forma clara, precisa e baseada em evidencias. "
            "Cite sempre o protocolo ou a fonte quando possivel. "
            "NUNCA prescrevera medicamentos diretamente; use sempre 'sugere-se avaliar "
            "com o medico responsavel'."
        )

        formatted = []
        for rec in records:
            context = self.anonymize_text(rec.get("context", ""))
            question = rec.get("question", "")
            answer = self.anonymize_text(rec.get("answer", ""))

            if not question or not answer:
                continue

            text = (
                f"### Instrucao:\n{system_instruction}\n\n"
                f"### Contexto:\n{context}\n\n"
                f"### Pergunta:\n{question}\n\n"
                f"### Resposta:\n{answer}"
            )
            formatted.append(
                {
                    "text": text,
                    "source_id": str(rec.get("id", "")),
                    "source": rec.get("source", ""),
                }
            )

        logger.info(f"Formatados {len(formatted)} exemplos para fine-tuning")
        return formatted

    def split_dataset(
        self,
        records: List,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple[List, List, List]:
        """
        Divide o dataset em conjuntos de treino, validacao e teste.

        Os dados nao sao embaralhados de forma aleatoria para garantir
        reprodutibilidade sem dependencia de seed.

        Returns:
            tupla (treino, validacao, teste)
        """
        n = len(records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = records[:n_train]
        val = records[n_train : n_train + n_val]
        test = records[n_train + n_val :]

        logger.info(
            f"Divisao do dataset: {len(train)} treino / {len(val)} validacao / {len(test)} teste"
        )
        return train, val, test

    def save_jsonl(self, records: List[Dict], filepath: str) -> None:
        """Salva uma lista de dicionarios em formato JSONL."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Salvo {len(records)} registros em {filepath}")

    def load_pubmedqa(
        self,
        fold: int = 0,
        include_synthetic: bool = True,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Carrega o dataset PubMedQA pre-processado (arquivos JSONL gerados
        pelo pubmedqa_converter.py) e opcionalmente concatena com o dataset
        sintetico original do projeto.

        O PubMedQA possui:
          - 450 exemplos de treino (fold0/train_set.json)
          -  50 exemplos de validacao (fold0/dev_set.json)
          - 500 exemplos de teste (test_set.json)

        Args:
            fold: indice do fold a usar (0-9); os arquivos devem ter sido
                  gerados com pubmedqa_converter.py usando o mesmo fold.
            include_synthetic: se True, concatena os 30 exemplos sinteticos
                  do projeto ao conjunto de treino (aumenta diversidade).

        Returns:
            tupla (train, val, test) com listas de dicionarios no formato
            padrao do projeto (campos: id, question, context, answer, type, source).
        """
        base = Path("data")
        train_path = base / "pubmedqa_train.jsonl"
        val_path = base / "pubmedqa_val.jsonl"
        test_path = base / "pubmedqa_test.jsonl"

        missing = [p for p in (train_path, val_path, test_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Arquivos PubMedQA nao encontrados: {missing}\n"
                "Execute: python src/pubmedqa_converter.py"
            )

        train = self.load_jsonl(str(train_path))
        val = self.load_jsonl(str(val_path))
        test = self.load_jsonl(str(test_path))

        if include_synthetic:
            synthetic_path = base / "synthetic_medical_qa.jsonl"
            if synthetic_path.exists():
                synthetic = self.load_jsonl(str(synthetic_path))
                train = train + synthetic
                logger.info(
                    f"Dataset sintetico concatenado ao treino: "
                    f"{len(synthetic)} registros adicionados."
                )

        logger.info(
            f"PubMedQA carregado: {len(train)} treino / "
            f"{len(val)} validacao / {len(test)} teste"
        )
        return train, val, test
