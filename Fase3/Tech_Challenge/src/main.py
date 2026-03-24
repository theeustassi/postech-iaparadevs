"""
Ponto de entrada principal - Assistente Medico Virtual HospitalIQ.

Tech Challenge - Fase 3 | FIAP Pos-Tech IA para Devs
Autor: Matheus Tassi Souza - RM367424

Este script demonstra o pipeline completo:
    1. Preprocessamento e preparacao do dataset
    2. Fine-tuning do LLM com LoRA (ou carregamento do modelo treinado)
    3. Criacao do assistente RAG com LangChain
    4. Execucao de consultas de exemplo
    5. Fluxo clinico com LangGraph

Variaveis de ambiente necessarias (opcional para modo local):
    GEMINI_API_KEY: chave da API do Google Gemini
                    (obtida em https://makersuite.google.com/app/apikey)

Uso:
    python src/main.py                  # pipeline completo
    python src/main.py --skip-finetune  # pula o fine-tuning (usa modelo base)
    python src/main.py --demo-only      # apenas demonstracao do assistente
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Permite imports relativos ao executar como script
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# Caminhos dos arquivos de dados
DATA_DIR = Path("data")
QA_DATASET = DATA_DIR / "pubmedqa_train.jsonl"
PROTOCOLS_FILE = DATA_DIR / "medical_protocols.txt"
PATIENT_RECORDS = DATA_DIR / "patient_records_sample.json"

RESULTS_DIR = Path("results")
VECTORSTORE_PATH = "results/modelos/vectorstore"
FINETUNED_MODEL_PATH = "results/modelos/finetuned_model"


def step_preprocess():
    """Etapa 1: Carrega, anonimiza e formata os dados PubMedQA para fine-tuning."""
    from preprocessing import MedicalDataPreprocessor

    logger.info("=== ETAPA 1: PREPROCESSAMENTO ===")
    preprocessor = MedicalDataPreprocessor()

    train_raw, val_raw, test_raw = preprocessor.load_pubmedqa(include_synthetic=False)

    # Aplica anonimizacao e formatacao instruction-following em cada split
    train = preprocessor.format_for_finetuning(train_raw)
    val   = preprocessor.format_for_finetuning(val_raw)
    test  = preprocessor.format_for_finetuning(test_raw)

    # Salva os splits processados
    preprocessor.save_jsonl(train, "results/data/train.jsonl")
    preprocessor.save_jsonl(val,   "results/data/val.jsonl")
    preprocessor.save_jsonl(test,  "results/data/test.jsonl")

    # Carrega e anonimiza registros de pacientes
    patients = preprocessor.load_patient_records(str(PATIENT_RECORDS))
    logger.info(f"Registros de pacientes anonimizados: {len(patients)}")

    return train, val, test, test_raw


def step_finetune(train_records, val_records, skip: bool = False):
    """
    Etapa 2: Fine-tuning do LLM com LoRA.

    Por padrao usa google/flan-t5-base que roda em CPU.
    Com skip=True, verifica se ja existe um modelo salvo; se nao,
    apenas inicializa o base model sem treinar.
    """
    from fine_tuning import FineTuningConfig, MedicalLLMFineTuner

    logger.info("=== ETAPA 2: FINE-TUNING ===")

    config = FineTuningConfig(
        model_name="google/flan-t5-base",
        output_dir=FINETUNED_MODEL_PATH,
        max_steps=50,   # Reduzido para demo rapido; aumentar para treinamento real
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        lora_r=8,
        lora_alpha=32,
    )

    tuner = MedicalLLMFineTuner(config)

    if skip:
        logger.info("Fine-tuning pulado (--skip-finetune ativo).")
        logger.info(
            "Para treinar: execute sem --skip-finetune ou com GPU disponivel."
        )
        return tuner

    if Path(FINETUNED_MODEL_PATH).exists():
        logger.info(f"Modelo fine-tunado encontrado em: {FINETUNED_MODEL_PATH}")
        try:
            tuner.load_finetuned(FINETUNED_MODEL_PATH)
            logger.info("Modelo carregado com sucesso.")
            return tuner
        except Exception as e:
            logger.warning(f"Nao foi possivel carregar o modelo salvo: {e}")

    logger.info(
        f"Iniciando fine-tuning com {len(train_records)} exemplos de treino..."
    )
    try:
        tuner.load_model()
        tuner.train(train_records, val_records)
        logger.info("Fine-tuning concluido.")
    except Exception as e:
        logger.warning(
            f"Fine-tuning nao executado (possivel falta de GPU/memoria): {e}\n"
            "Continuando com o modelo base sem fine-tuning."
        )

    return tuner


def step_build_assistant():
    """
    Etapa 3: Cria o assistente RAG com LangChain.

    Carrega a base vetorial existente ou cria uma nova a partir dos documentos.
    """
    from assistant import MedicalAssistant

    logger.info("=== ETAPA 3: CONSTRUCAO DO ASSISTENTE RAG ===")

    api_key = os.getenv("GEMINI_API_KEY", "")
    assistant = MedicalAssistant(api_key=api_key)

    if Path(VECTORSTORE_PATH).exists():
        logger.info("Base vetorial existente encontrada. Carregando...")
        assistant.load_knowledge_base(VECTORSTORE_PATH)
    else:
        logger.info("Criando base vetorial a partir dos documentos...")
        assistant.build_knowledge_base(
            [str(PROTOCOLS_FILE), str(QA_DATASET)]
        )
        assistant.save_knowledge_base(VECTORSTORE_PATH)

    assistant.build_rag_chain()
    logger.info("Assistente RAG pronto.")
    return assistant


def step_demo_queries(assistant, security_modules):
    """Etapa 4: Demonstra consultas ao assistente."""
    from security import AuditLogger, InputValidator

    logger.info("=== ETAPA 4: DEMONSTRACAO DE CONSULTAS ===")

    audit_logger = AuditLogger()
    validator = InputValidator()

    sample_queries = [
        {
            "query": "Qual e o bundle de 1 hora para sepse?",
            "patient_context": "Paciente com febre, hipotensao e taquicardia. Suspeita de infeccao.",
        },
        {
            "query": "Como estratificar a gravidade da pneumonia adquirida na comunidade?",
            "patient_context": None,
        },
        {
            "query": "Quando indicar anticoagulacao na fibrilacao atrial?",
            "patient_context": "Paciente de 72 anos com FA cronica, hipertensao e diabetes.",
        },
    ]

    for i, sample in enumerate(sample_queries, 1):
        query = sample["query"]
        patient_ctx = sample.get("patient_context")

        print(f"\n{'='*60}")
        print(f"Consulta {i}: {query}")

        # Valida entrada
        valid, error_msg = validator.validate_query(query)
        if not valid:
            print(f"Entrada invalida: {error_msg}")
            continue

        try:
            result = assistant.ask(query, patient_context=patient_ctx)

            print(f"\nResposta:")
            print(result["response"])
            print(f"\nFontes: {', '.join(result['sources'])}")
            print(f"\n{result['safety_disclaimer']}")

            # Registra na auditoria
            audit_logger.log_query(
                patient_id="DEMO-001",
                query=query,
                response=result["response"],
                severity="LOW",
                sources=result["sources"],
            )

        except Exception as e:
            logger.error(f"Erro na consulta {i}: {e}")


def step_langgraph_demo(assistant):
    """Etapa 5: Demonstra o fluxo clinico com LangGraph."""
    from langgraph_flows import build_clinical_workflow

    logger.info("=== ETAPA 5: FLUXO CLINICO COM LANGGRAPH ===")

    app = build_clinical_workflow(assistant=assistant)

    # Caso de teste 1: paciente de alta gravidade
    state_alta_gravidade = {
        "patient_id": "PAC-DEMO-001",
        "patient_info": {
            "age": 67,
            "sex": "M",
            "chief_complaint": "dor no peito com irradiacao para braco esquerdo ha 1 hora",
            "vital_signs": {
                "systolic_bp": 90,
                "diastolic_bp": 60,
                "heart_rate": 115,
                "respiratory_rate": 22,
                "temperature": 36.8,
                "spo2": 94,
            },
            "comorbidities": ["hipertensao", "diabetes tipo 2"],
            "current_medications": ["metformina", "enalapril"],
            "allergies": [],
            "pending_exams": [],
        },
        "query": "Qual a conduta para esse paciente?",
        "severity": "",
        "severity_reason": "",
        "pending_exams": [],
        "clinical_suggestions": "",
        "critical_alerts": [],
        "safety_check_passed": False,
        "final_response": "",
        "audit_trail": [],
        "error": None,
    }

    print("\n" + "=" * 60)
    print("FLUXO LANGGRAPH - Caso 1: Alta Gravidade")
    print("=" * 60)

    try:
        result = app.invoke(state_alta_gravidade)
        print(result["final_response"])
        print(f"\nAudit trail: {len(result['audit_trail'])} etapas registradas")
    except Exception as e:
        logger.error(f"Erro no fluxo LangGraph (caso 1): {e}")

    # Caso de teste 2: paciente de baixa gravidade
    state_baixa_gravidade = {
        "patient_id": "PAC-DEMO-002",
        "patient_info": {
            "age": 35,
            "sex": "F",
            "chief_complaint": "tosse seca ha 5 dias, sem febre",
            "vital_signs": {
                "systolic_bp": 118,
                "diastolic_bp": 76,
                "heart_rate": 78,
                "respiratory_rate": 16,
                "temperature": 37.1,
                "spo2": 98,
            },
            "comorbidities": [],
            "current_medications": [],
            "allergies": [],
            "pending_exams": [],
        },
        "query": "Quais exames solicitar e qual conduta?",
        "severity": "",
        "severity_reason": "",
        "pending_exams": [],
        "clinical_suggestions": "",
        "critical_alerts": [],
        "safety_check_passed": False,
        "final_response": "",
        "audit_trail": [],
        "error": None,
    }

    print("\n" + "=" * 60)
    print("FLUXO LANGGRAPH - Caso 2: Baixa Gravidade")
    print("=" * 60)

    try:
        result = app.invoke(state_baixa_gravidade)
        print(result["final_response"])
    except Exception as e:
        logger.error(f"Erro no fluxo LangGraph (caso 2): {e}")


def step_evaluate(test_records, tuner):
    """Etapa 6: Avalia o modelo com metricas automaticas."""
    from evaluation import ModelEvaluator

    logger.info("=== ETAPA 6: AVALIACAO DO MODELO ===")

    evaluator = ModelEvaluator()

    # Usa apenas os primeiros 10 exemplos para demo rapida
    test_sample = test_records[:10] if len(test_records) > 10 else test_records

    if tuner.model is None:
        logger.info(
            "Modelo nao carregado. Avaliacao de seguranca apenas (sem BLEU/ROUGE)."
        )
        # Avalia apenas as respostas de seguranca com respostas fixas de demo
        demo_responses = [rec.get("answer", "") for rec in test_sample]
        safety = evaluator.safety_audit(demo_responses)
        print(f"\nSafety compliance: {safety.get('safety_compliance_rate', 0):.1%}")
        return

    metrics = evaluator.evaluate_model(
        test_records=test_sample,
        generate_fn=tuner.generate,
        output_path="results/evaluation_results.json",
    )
    evaluator.print_summary(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Assistente Medico Virtual HospitalIQ"
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Pula o fine-tuning e usa o modelo base (mais rapido para demo)",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Apenas demonstra o assistente (sem fine-tuning ou avaliacao)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TECH CHALLENGE FASE 3 - HospitalIQ")
    print("Assistente Medico Virtual com LLM + LangChain + LangGraph")
    print("Autor: Matheus Tassi Souza - RM367424")
    print("=" * 60 + "\n")

    # Garante que os diretorios de saida existem
    for d in ["results/data", "results/modelos", "results/logs", "results/graficos"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.demo_only:
        assistant = step_build_assistant()
        step_demo_queries(assistant, None)
        step_langgraph_demo(assistant)
        return

    # Pipeline completo
    train, val, test, raw_records = step_preprocess()
    tuner = step_finetune(train, val, skip=args.skip_finetune)
    assistant = step_build_assistant()
    step_demo_queries(assistant, None)
    step_langgraph_demo(assistant)
    step_evaluate(raw_records, tuner)

    print("\nPipeline concluido com sucesso.")
    print(f"Resultados salvos em: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
