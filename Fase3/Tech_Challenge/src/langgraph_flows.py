"""
Fluxos de decisao clinica com LangGraph.

Este modulo implementa um pipeline automatizado de triagem e apoio medico
usando um grafo de estados (StateGraph) do LangGraph.

Fluxo do grafo:
    [INICIO]
        |
    patient_intake       <- Estrutura os dados do paciente
        |
    triage               <- Avalia gravidade (LOW/MEDIUM/HIGH/CRITICAL)
        |
    check_exams          <- Verifica e sugere exames pendentes
        |
    [decisao por gravidade]
       / \
 CRITICAL  outros
    |         |
critical_   clinical_
 alert      reasoning  <- Consulta o assistente RAG
     \      /
    safety_check        <- Filtra prescricoes diretas, adiciona disclaimers
        |
    generate_response   <- Formata a resposta final
        |
    audit_logging       <- Registra todo o fluxo para auditoria
        |
    [FIM]

Cada no (node) recebe o estado completo e retorna apenas os campos alterados.
O campo 'audit_trail' usa operator.add para acumular entradas (nao substitui).
"""

import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
import operator
from typing import TypedDict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Definicao do estado do fluxo
# ------------------------------------------------------------------

class PatientState(TypedDict):
    """
    Estado completo do fluxo de atendimento.

    Cada no do grafo le e atualiza campos deste estado.
    O campo audit_trail e acumulativo (operator.add) para registrar
    todas as etapas sem sobrescrever entradas anteriores.
    """

    patient_id: str
    patient_info: Dict[str, Any]
    query: str
    severity: str                                      # LOW | MEDIUM | HIGH | CRITICAL
    severity_reason: str
    pending_exams: List[str]
    clinical_suggestions: str
    critical_alerts: List[str]
    safety_check_passed: bool
    final_response: str
    audit_trail: Annotated[List[Dict[str, Any]], operator.add]
    error: Optional[str]


# ------------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------------

def _audit_entry(step: str, state: PatientState, **extra) -> Dict[str, Any]:
    """Cria uma entrada de auditoria padrao para o passo atual."""
    entry = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "patient_id": state.get("patient_id", "desconhecido"),
        "severity": state.get("severity", "N/A"),
    }
    entry.update(extra)
    return entry


# ------------------------------------------------------------------
# Nos do grafo
# ------------------------------------------------------------------

def patient_intake_node(state: PatientState) -> Dict[str, Any]:
    """
    Estrutura e normaliza os dados do paciente recebidos na entrada.

    Garante que todos os campos esperados existam no estado,
    prevenindo erros em nos subsequentes.
    """
    raw = state.get("patient_info", {})
    query = state.get("query", "")

    structured = {
        "age": raw.get("age", "nao informado"),
        "sex": raw.get("sex", "nao informado"),
        "chief_complaint": raw.get("chief_complaint", query),
        "vital_signs": raw.get("vital_signs", {}),
        "allergies": raw.get("allergies", []),
        "current_medications": raw.get("current_medications", []),
        "comorbidities": raw.get("comorbidities", []),
        "pending_exams": raw.get("pending_exams", []),
        "notes": raw.get("notes", ""),
    }

    audit = _audit_entry(
        "patient_intake",
        state,
        chief_complaint=structured["chief_complaint"],
    )
    return {
        "patient_info": structured,
        "audit_trail": [audit],
    }


def triage_node(state: PatientState) -> Dict[str, Any]:
    """
    Classifica a gravidade do paciente com base em sinais vitais e queixa.

    Criterios de triagem inspirados no Protocolo de Manchester:
        CRITICAL: palavras-chave de emergencia ou sinais vitais criticos
        HIGH:     palavras-chave de urgencia ou sinais vitais alterados
        MEDIUM:   sinais vitais leve/moderadamente alterados
        LOW:      sem criterios de alarme
    """
    patient_info = state.get("patient_info", {})
    complaint = patient_info.get("chief_complaint", "").lower()
    vs = patient_info.get("vital_signs", {})

    CRITICAL_KEYWORDS = [
        "parada cardiaca",
        "parada respiratoria",
        "pcr",
        "infarto",
        "avc",
        "convulsao",
        "inconsciencia",
        "sangramento grave",
        "choque",
        "sepse",
        "dispneia grave",
        "edema agudo",
    ]
    HIGH_KEYWORDS = [
        "dor no peito",
        "dor toracica",
        "falta de ar",
        "febre alta",
        "confusao mental",
        "trauma",
        "fratura",
        "hipotensao",
        "taquicardia",
    ]

    severity = "LOW"
    reason = "Sem criterios de urgencia ou emergencia identificados"

    if any(k in complaint for k in CRITICAL_KEYWORDS):
        severity = "CRITICAL"
        reason = "Queixa indica emergencia medica imediata"
    elif any(k in complaint for k in HIGH_KEYWORDS):
        severity = "HIGH"
        reason = "Queixa indica urgencia clinica"

    # Avalia sinais vitais
    spo2 = float(vs.get("spo2", 100))
    pas = float(vs.get("systolic_bp", 120))
    fc = float(vs.get("heart_rate", 80))
    temp = float(vs.get("temperature", 36.5))

    if spo2 < 90 or pas < 80 or fc > 150 or fc < 30:
        severity = "CRITICAL"
        reason = (
            f"Sinais vitais criticos: SpO2={spo2:.0f}%, PAS={pas:.0f}mmHg, "
            f"FC={fc:.0f}bpm"
        )
    elif spo2 < 94 or pas < 95 or fc > 120 or temp > 39.0:
        if severity != "CRITICAL":
            severity = "HIGH"
            reason = (
                f"Sinais vitais alterados: SpO2={spo2:.0f}%, PAS={pas:.0f}mmHg, "
                f"T={temp:.1f}C"
            )
    elif temp > 37.8 or fc > 100:
        if severity == "LOW":
            severity = "MEDIUM"
            reason = "Sinais vitais levemente alterados"

    audit = _audit_entry("triage", state, severity_assigned=severity, reason=reason)
    return {
        "severity": severity,
        "severity_reason": reason,
        "audit_trail": [audit],
    }


def check_exams_node(state: PatientState) -> Dict[str, Any]:
    """
    Verifica exames ja solicitados e sugere exames adicionais
    com base na gravidade e na queixa principal.
    """
    patient_info = state.get("patient_info", {})
    severity = state.get("severity", "LOW")
    complaint = patient_info.get("chief_complaint", "").lower()

    # Exames ja registrados no prontuario
    existing = list(patient_info.get("pending_exams", []))

    # Sugestoes baseadas na gravidade
    suggested: List[str] = []
    if severity in ("HIGH", "CRITICAL"):
        suggested.extend(
            ["Hemograma completo", "PCR", "Lactato serico", "Gasometria arterial"]
        )

    # Sugestoes por queixa
    if any(k in complaint for k in ("dor no peito", "dor toracica", "infarto")):
        suggested.extend(["Troponina de alta sensibilidade", "ECG 12 derivacoes"])
    if any(k in complaint for k in ("falta de ar", "dispneia", "tosse")):
        suggested.extend(["Radiografia de torax", "Oximetria de pulso continua"])
    if any(k in complaint for k in ("febre", "sepse", "infeccao")):
        suggested.extend(["Hemocultura (2 pares)", "Urina I + urocultura", "Procalcitonina"])
    if any(k in complaint for k in ("avc", "fraqueza", "alteracao de consciencia")):
        suggested.extend(["TC de cranio sem contraste", "Glicemia capilar urgente"])
    if any(k in complaint for k in ("dor abdominal", "abdome")):
        suggested.extend(["Ultrassom de abdome total", "Amilase", "Lipase"])

    # Unifica sem duplicatas e preserva a ordem
    all_exams = existing[:]
    for exam in suggested:
        if exam not in all_exams:
            all_exams.append(exam)

    audit = _audit_entry("check_exams", state, exams_total=len(all_exams))
    return {
        "pending_exams": all_exams,
        "audit_trail": [audit],
    }


def critical_alert_node(state: PatientState) -> Dict[str, Any]:
    """
    Gera e registra um alerta critico para acionamento da equipe de emergencia.
    Este no e ativado apenas quando severity == CRITICAL.
    """
    patient_id = state.get("patient_id", "DESCONHECIDO")
    reason = state.get("severity_reason", "Criterio nao especificado")
    vs = state.get("patient_info", {}).get("vital_signs", {})

    alert = (
        f"[ALERTA CRITICO] Paciente {patient_id} requer atencao IMEDIATA.\n"
        f"Motivo: {reason}\n"
        f"Sinais vitais: {vs}\n"
        f"ACAO REQUERIDA: Acionar equipe de emergencia AGORA."
    )

    logger.critical(alert)

    audit = _audit_entry("critical_alert", state, alert_issued=True)
    return {
        "critical_alerts": [alert],
        "audit_trail": [audit],
    }


def clinical_reasoning_node(
    state: PatientState,
    assistant=None,
) -> Dict[str, Any]:
    """
    Consulta o assistente RAG para obter sugestoes clinicas baseadas em protocolos.

    Se o assistente nao estiver disponivel, recorre a sugestoes genericas
    baseadas na gravidade.
    """
    patient_info = state.get("patient_info", {})
    query = state.get("query", "")
    severity = state.get("severity", "LOW")
    exams = state.get("pending_exams", [])

    patient_context = (
        f"Paciente com {patient_info.get('age', '?')} anos, sexo {patient_info.get('sex', '?')}. "
        f"Gravidade: {severity}. "
        f"Queixa: {patient_info.get('chief_complaint', query)}. "
        f"Comorbidades: {', '.join(patient_info.get('comorbidities', ['nenhuma']))}. "
        f"Medicamentos em uso: {', '.join(patient_info.get('current_medications', ['nenhum']))}. "
        f"Alergias: {', '.join(patient_info.get('allergies', ['nenhuma']))}. "
        f"Exames pendentes: {', '.join(exams[:6]) if exams else 'nenhum'}."
    )

    suggestions = ""
    if assistant is not None:
        try:
            result = assistant.ask(query, patient_context=patient_context)
            suggestions = result.get("response", "")
        except Exception as e:
            logger.warning(f"Assistente RAG indisponivel: {e}. Usando sugestoes genericas.")

    if not suggestions:
        suggestions = _generic_suggestions(
            severity, patient_info.get("chief_complaint", query)
        )

    audit = _audit_entry("clinical_reasoning", state, used_rag=assistant is not None)
    return {
        "clinical_suggestions": suggestions,
        "audit_trail": [audit],
    }


def _generic_suggestions(severity: str, chief_complaint: str) -> str:
    """Sugestoes genericas quando o assistente RAG nao esta disponivel."""
    base = f"Avaliacao para '{chief_complaint}' | Gravidade: {severity}\n\n"
    if severity == "CRITICAL":
        return (
            base
            + "- Acionar equipe de emergencia IMEDIATAMENTE.\n"
            + "- Monitoramento continuo de sinais vitais.\n"
            + "- Estabilizacao hemodinamica prioritaria.\n"
            + "- Preparar acesso venoso central se necessario."
        )
    if severity == "HIGH":
        return (
            base
            + "- Avaliacao medica em ate 30 minutos.\n"
            + "- Monitorizar sinais vitais continuamente.\n"
            + "- Instituir acesso venoso periférico.\n"
            + "- Aguardar resultado dos exames prioritarios."
        )
    if severity == "MEDIUM":
        return (
            base
            + "- Avaliacao medica em ate 2 horas.\n"
            + "- Monitorar evolucao dos sintomas.\n"
            + "- Solicitar exames conforme indicacao clinica."
        )
    return (
        base
        + "- Avaliacao medica de rotina.\n"
        + "- Orientar sobre sinais de alerta para retorno imediato."
    )


def safety_check_node(state: PatientState) -> Dict[str, Any]:
    """
    Valida as sugestoes clinicas removendo qualquer prescricao direta
    e adicionando os disclaimers obrigatorios.

    Termos proibidos: qualquer forma de prescricao direta de medicamentos
    sem a mediacao do medico responsavel.
    """
    suggestions = state.get("clinical_suggestions", "")
    severity = state.get("severity", "LOW")

    FORBIDDEN_TERMS = [
        "tome ",
        "tomar ",
        "use ",
        "usar ",
        "administre ",
        "administrar ",
        "prescrev",
        "mg/dia",
        "mg/kg",
        "comprimido por dia",
        "ampola ev",
        "infusao de ",
    ]

    has_forbidden = any(term in suggestions.lower() for term in FORBIDDEN_TERMS)

    if has_forbidden:
        # Filtra as linhas com prescricao direta
        filtered_lines = [
            line
            for line in suggestions.split("\n")
            if not any(t in line.lower() for t in FORBIDDEN_TERMS)
        ]
        suggestions = (
            "[AVISO: CONTEUDO COM PRESCRICAO DIRETA REMOVIDO POR SEGURANCA]\n\n"
            "O sistema identificou e removeu sugestoes de prescricao direta. "
            "Consulte o medico responsavel para definir a conduta farmacologica.\n\n"
            + "\n".join(filtered_lines)
        )
        logger.warning(
            f"Prescricao direta detectada e removida para paciente "
            f"{state.get('patient_id', 'desconhecido')}"
        )

    # Adiciona disclaimer obrigatorio
    disclaimer = (
        "\n\n---\n"
        "IMPORTANTE: Estas sugestoes sao de carater informativo e de apoio clinico. "
        "A decisao terapeutica final e de responsabilidade exclusiva do medico assistente. "
        "Este sistema nao substitui a avaliacao clinica presencial."
    )

    if severity == "CRITICAL":
        disclaimer += (
            "\nCASO CRITICO: Acione imediatamente a equipe de emergencia. "
            "Nao aguarde o resultado de exames para iniciar o suporte basico."
        )

    audit = _audit_entry("safety_check", state, content_blocked=has_forbidden)
    return {
        "clinical_suggestions": suggestions + disclaimer,
        "safety_check_passed": True,
        "audit_trail": [audit],
    }


def generate_response_node(state: PatientState) -> Dict[str, Any]:
    """
    Formata a resposta final consolidada para o profissional de saude.
    """
    patient_id = state.get("patient_id", "N/A")
    severity = state.get("severity", "LOW")
    reason = state.get("severity_reason", "")
    exams = state.get("pending_exams", [])
    suggestions = state.get("clinical_suggestions", "")
    alerts = state.get("critical_alerts", [])

    SEVERITY_LABELS = {
        "LOW": "Baixa",
        "MEDIUM": "Media",
        "HIGH": "Alta",
        "CRITICAL": "CRITICA - EMERGENCIA",
    }
    label = SEVERITY_LABELS.get(severity, severity)

    exams_section = ""
    if exams:
        exams_list = "\n".join(f"  - {e}" for e in exams[:10])
        exams_section = f"\nExames recomendados:\n{exams_list}\n"

    alerts_section = ""
    if alerts:
        alerts_section = "\n" + "\n".join(alerts) + "\n"

    response = (
        "=== ASSISTENTE MEDICO VIRTUAL - HospitalIQ ===\n"
        f"Paciente: {patient_id}  |  Gravidade: {label}\n"
        f"Avaliacao: {reason}\n"
        f"{exams_section}"
        f"{alerts_section}\n"
        "--- Condutas Sugeridas ---\n"
        f"{suggestions}"
    )

    audit = _audit_entry("generate_response", state)
    return {
        "final_response": response,
        "audit_trail": [audit],
    }


def audit_logging_node(state: PatientState) -> Dict[str, Any]:
    """
    Registra o audit trail completo no logger para rastreabilidade e auditoria.
    Em producao, este node salvaria o registro em banco de dados ou sistema de logs.
    """
    patient_id = state.get("patient_id", "desconhecido")
    trail = state.get("audit_trail", [])
    steps = [entry.get("step", "?") for entry in trail]

    logger.info(
        f"Fluxo concluido | Paciente: {patient_id} | "
        f"Gravidade: {state.get('severity', 'N/A')} | "
        f"Etapas: {steps}"
    )

    audit = _audit_entry("audit_logging", state, total_steps=len(trail) + 1)
    return {"audit_trail": [audit]}


# ------------------------------------------------------------------
# Roteamento condicional
# ------------------------------------------------------------------

def route_by_severity(state: PatientState) -> str:
    """
    Determina o proximo no com base na gravidade classificada na triagem.

    CRITICAL -> critical_alert (para acionamento imediato da equipe)
    outros   -> clinical_reasoning (para sugestoes baseadas em protocolo)
    """
    if state.get("severity") == "CRITICAL":
        return "critical_alert"
    return "clinical_reasoning"


# ------------------------------------------------------------------
# Construcao do grafo
# ------------------------------------------------------------------

def build_clinical_workflow(assistant=None):
    """
    Compila e retorna o grafo de fluxo clinico do HospitalIQ.

    Args:
        assistant: instancia de MedicalAssistant para consulta RAG.
                   Se None, o no clinical_reasoning usa sugestoes genericas.

    Returns:
        grafo compilado pronto para receber estados via .invoke()

    Exemplo de uso:
        app = build_clinical_workflow(assistant=assistant)
        result = app.invoke({
            "patient_id": "PAC-001",
            "patient_info": {"chief_complaint": "dor no peito", "age": 65},
            "query": "Qual a conduta?",
            "severity": "",
            "severity_reason": "",
            "pending_exams": [],
            "clinical_suggestions": "",
            "critical_alerts": [],
            "safety_check_passed": False,
            "final_response": "",
            "audit_trail": [],
            "error": None,
        })
        print(result["final_response"])
    """
    import functools

    from langgraph.graph import END, StateGraph

    # Injeta o assistente no no de raciocinio clinico via partial
    cr_node = functools.partial(clinical_reasoning_node, assistant=assistant)

    graph = StateGraph(PatientState)

    # Registra os nos
    graph.add_node("patient_intake", patient_intake_node)
    graph.add_node("triage", triage_node)
    graph.add_node("check_exams", check_exams_node)
    graph.add_node("critical_alert", critical_alert_node)
    graph.add_node("clinical_reasoning", cr_node)
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("audit_logging", audit_logging_node)

    # Define o ponto de entrada e as transicoes
    graph.set_entry_point("patient_intake")
    graph.add_edge("patient_intake", "triage")
    graph.add_edge("triage", "check_exams")

    # Roteamento condicional apos verificacao de exames
    graph.add_conditional_edges(
        "check_exams",
        route_by_severity,
        {
            "critical_alert": "critical_alert",
            "clinical_reasoning": "clinical_reasoning",
        },
    )

    graph.add_edge("critical_alert", "safety_check")
    graph.add_edge("clinical_reasoning", "safety_check")
    graph.add_edge("safety_check", "generate_response")
    graph.add_edge("generate_response", "audit_logging")
    graph.add_edge("audit_logging", END)

    return graph.compile()
