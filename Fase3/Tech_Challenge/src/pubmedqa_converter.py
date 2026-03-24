"""
Conversor do dataset PubMedQA para o formato JSONL do projeto.

PubMedQA (Jin et al., 2019) e um dataset de pergunta e resposta biomedica
extraido do PubMed. Cada registro contem:
  - QUESTION: pergunta de pesquisa biomedica
  - CONTEXTS: paragrafos do abstract (secoes como BACKGROUND, RESULTS, etc.)
  - LABELS: nome de cada secao do abstract
  - LONG_ANSWER: resposta longa (conclusao do paper)
  - final_decision: yes / no / maybe
  - MESHES: termos MeSH relacionados
  - YEAR: ano de publicacao

O script converte para o formato JSONL do projeto:
  {"id", "question", "context", "answer", "type", "source"}

Uso:
    python src/pubmedqa_converter.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PUBMEDQA_DIR = Path("data/pubmedqa/data")
OUTPUT_DIR = Path("data")


def _build_context(contexts: List[str], labels: Optional[List[str]] = None) -> str:
    """
    Junta os paragrafos do abstract em um unico texto estruturado.

    Cada secao e prefixada com seu label (BACKGROUND, RESULTS, etc.)
    quando disponivel.
    """
    parts = []
    for i, ctx in enumerate(contexts):
        if labels and i < len(labels):
            parts.append(f"[{labels[i]}] {ctx.strip()}")
        else:
            parts.append(ctx.strip())
    return "\n\n".join(parts)


def _build_answer(long_answer: str, decision: str) -> str:
    """
    Monta a resposta completa com a conclusao e a decisao final.

    O field 'final_decision' (yes/no/maybe) e incluido para que o modelo
    aprenda o formato de resposta estruturada.
    """
    decision_map = {
        "yes": "Sim",
        "no": "Nao",
        "maybe": "Inconclusivo",
    }
    label = decision_map.get(decision.lower(), decision)
    return f"{long_answer.strip()}\n\nDecisao: {label} ({decision})"


def convert_split(
    input_path: Path,
    split_name: str,
    max_records: Optional[int] = None,
) -> List[Dict]:
    """
    Converte um arquivo JSON do PubMedQA para lista de dicionarios no
    formato JSONL do projeto.

    Args:
        input_path: caminho para o arquivo JSON (ex: pqal_fold0/train_set.json)
        split_name: nome do split para usar no campo 'source'
        max_records: limite de registros a converter (None = todos)

    Returns:
        lista de dicionarios prontos para salvar em JSONL
    """
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for pmid, item in raw.items():
        if max_records and len(records) >= max_records:
            break

        question = item.get("QUESTION", "").strip()
        contexts = item.get("CONTEXTS", [])
        labels = item.get("LABELS", [])
        long_answer = item.get("LONG_ANSWER", "").strip()
        decision = item.get("final_decision", "").strip()
        meshes = item.get("MESHES", [])
        year = item.get("YEAR", "")

        if not question or not long_answer:
            continue

        records.append({
            "id": f"pubmedqa_{pmid}",
            "question": question,
            "context": _build_context(contexts, labels),
            "answer": _build_answer(long_answer, decision),
            "type": "pubmedqa_labeled",
            "source": f"PubMedQA-L ({split_name}, PMID {pmid})",
            "metadata": {
                "pmid": pmid,
                "year": year,
                "final_decision": decision,
                "meshes": meshes[:5],
            },
        })

    return records


def save_jsonl(records: List[Dict], output_path: Path) -> None:
    """Salva lista de registros em formato JSONL (um JSON por linha)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Salvo: {output_path} ({len(records)} registros)")


def run_conversion(fold: int = 0) -> Dict[str, Path]:
    """
    Executa a conversao completa usando o fold especificado.

    Gera tres arquivos JSONL:
      - data/pubmedqa_train.jsonl  (treino: fold0/train_set.json = 450 registros)
      - data/pubmedqa_val.jsonl    (validacao: fold0/dev_set.json = 50 registros)
      - data/pubmedqa_test.jsonl   (teste: test_set.json = 500 registros)

    Args:
        fold: indice do fold de cross-validation a usar como treino/val (0-9)

    Returns:
        dicionario com os caminhos dos arquivos gerados
    """
    fold_dir = PUBMEDQA_DIR / f"pqal_fold{fold}"
    test_path = PUBMEDQA_DIR / "test_set.json"

    if not fold_dir.exists():
        raise FileNotFoundError(
            f"Fold nao encontrado: {fold_dir}\n"
            "Execute primeiro: cd data/pubmedqa/preprocess && python split_dataset.py pqal"
        )

    train_records = convert_split(fold_dir / "train_set.json", f"fold{fold}_train")
    val_records = convert_split(fold_dir / "dev_set.json", f"fold{fold}_dev")
    test_records = convert_split(test_path, "test")

    out_train = OUTPUT_DIR / "pubmedqa_train.jsonl"
    out_val = OUTPUT_DIR / "pubmedqa_val.jsonl"
    out_test = OUTPUT_DIR / "pubmedqa_test.jsonl"

    save_jsonl(train_records, out_train)
    save_jsonl(val_records, out_val)
    save_jsonl(test_records, out_test)

    print(f"Conversao concluida:")
    print(f"  Treino:    {len(train_records):>4} registros -> {out_train}")
    print(f"  Validacao: {len(val_records):>4} registros -> {out_val}")
    print(f"  Teste:     {len(test_records):>4} registros -> {out_test}")

    return {"train": out_train, "val": out_val, "test": out_test}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_conversion(fold=0)
