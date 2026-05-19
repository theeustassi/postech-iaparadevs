"""
Módulo de fusão multimodal: combina resultados de vídeo e áudio para
gerar uma avaliação de risco clínico consolidada.

A fusão é realizada por ponderação configurável de cada modalidade,
permitindo ajuste conforme o tipo de atendimento. O resultado final
é um RiscoMultimodal com pontuação unificada, nível de prioridade
e recomendações de conduta.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from audio_analysis import ResultadoAudio, TipoConsulta
from video_analysis import ResultadoVideo, TipoVideo

logger = logging.getLogger(__name__)


class NivelPrioridade(str, Enum):
    VERDE = "VERDE"       # Risco baixo - acompanhamento rotineiro
    AMARELO = "AMARELO"   # Risco moderado - revisão em até 48h
    LARANJA = "LARANJA"   # Risco alto - revisão em até 24h
    VERMELHO = "VERMELHO" # Risco crítico - intervenção imediata


@dataclass
class RiscoMultimodal:
    pontuacao_video: float
    pontuacao_audio: float
    pontuacao_fusao: float          # Score ponderado final (0.0 - 1.0)
    nivel_prioridade: NivelPrioridade
    alertas_criticos: List[str] = field(default_factory=list)
    alertas_moderados: List[str] = field(default_factory=list)
    recomendacoes: List[str] = field(default_factory=list)
    resumo_executivo: str = ""


# Tabela de pesos por tipo de atendimento
# (peso_video, peso_audio)
_PESOS_POR_TIPO: Dict[str, tuple] = {
    "cirurgia":            (0.75, 0.25),
    "consulta":            (0.40, 0.60),
    "fisioterapia":        (0.70, 0.30),
    "triagem_violencia":   (0.35, 0.65),
    "pre_natal":           (0.30, 0.70),
    "pos_parto":           (0.25, 0.75),
    "ginecologica":        (0.40, 0.60),
    "padrao":              (0.50, 0.50),
}


def _nivel_de_pontuacao(pontuacao: float) -> NivelPrioridade:
    if pontuacao < 0.20:
        return NivelPrioridade.VERDE
    if pontuacao < 0.45:
        return NivelPrioridade.AMARELO
    if pontuacao < 0.70:
        return NivelPrioridade.LARANJA
    return NivelPrioridade.VERMELHO


def _recomendacoes_por_nivel(
    nivel: NivelPrioridade,
    resultado_video: Optional[ResultadoVideo],
    resultado_audio: Optional[ResultadoAudio],
) -> List[str]:
    recomendacoes = []

    if nivel == NivelPrioridade.VERDE:
        recomendacoes.append("Seguimento ambulatorial conforme protocolo regular.")
        return recomendacoes

    if nivel == NivelPrioridade.AMARELO:
        recomendacoes.append("Agendar revisão em até 48 horas.")
        recomendacoes.append("Notificar médico responsável pelo caso.")

    if nivel == NivelPrioridade.LARANJA:
        recomendacoes.append("Acionar médico de plantão em até 24 horas.")
        recomendacoes.append("Registrar ocorrência no prontuário eletrônico.")
        recomendacoes.append("Considerar internação observacional.")

    if nivel == NivelPrioridade.VERMELHO:
        recomendacoes.append("INTERVENÇÃO IMEDIATA NECESSÁRIA.")
        recomendacoes.append("Acionar equipe médica agora.")
        recomendacoes.append("Registrar ocorrência crítica no sistema hospitalar.")

    if resultado_audio is not None:
        if resultado_audio.pontuacao_violencia > 0.4:
            recomendacoes.append(
                "Acionar assistente social e protocolo de violência doméstica (Lei Maria da Penha)."
            )
        if resultado_audio.pontuacao_depressao > 0.4:
            recomendacoes.append(
                "Aplicar Escala de Depressão Pós-Parto de Edimburgo (EPDS)."
            )
        if resultado_audio.pontuacao_ansiedade > 0.4:
            recomendacoes.append("Encaminhar para avaliação psicológica especializada.")

    if resultado_video is not None and resultado_video.tipo == TipoVideo.CIRURGIA:
        if resultado_video.pontuacao_risco_maxima > 0.5:
            recomendacoes.append(
                "Revisar imagens cirúrgicas com cirurgião sênior. "
                "Confirmar hemostasia adequada."
            )

    return recomendacoes


class FusorMultimodal:
    """
    Combina resultados de vídeo e áudio em uma avaliação de risco unificada.

    A fusão segue a estratégia de combinação linear ponderada, onde os pesos
    são definidos por tipo de atendimento (vídeo geralmente tem maior peso em
    procedimentos cirúrgicos; áudio tem maior peso em consultas de saúde mental).
    """

    def fundir(
        self,
        resultado_video: Optional[ResultadoVideo],
        resultado_audio: Optional[ResultadoAudio],
        tipo_atendimento: str = "padrao",
    ) -> RiscoMultimodal:
        """
        Realiza a fusão entre os resultados de vídeo e áudio.

        Se apenas uma modalidade estiver disponível, utiliza apenas ela com
        peso total. Se nenhuma estiver disponível, retorna risco zero.
        """
        if resultado_video is None and resultado_audio is None:
            logger.warning("Nenhuma modalidade disponível para fusão.")
            return RiscoMultimodal(
                pontuacao_video=0.0,
                pontuacao_audio=0.0,
                pontuacao_fusao=0.0,
                nivel_prioridade=NivelPrioridade.VERDE,
                resumo_executivo="Nenhuma modalidade disponível para análise.",
            )

        chave = tipo_atendimento.lower().replace(" ", "_")
        peso_video, peso_audio = _PESOS_POR_TIPO.get(chave, _PESOS_POR_TIPO["padrao"])

        # Ajusta pesos quando so uma modalidade esta disponivel
        if resultado_video is None:
            peso_video, peso_audio = 0.0, 1.0
        elif resultado_audio is None:
            peso_video, peso_audio = 1.0, 0.0

        p_video = resultado_video.pontuacao_risco_media if resultado_video else 0.0
        p_audio = resultado_audio.pontuacao_risco_geral if resultado_audio else 0.0

        pontuacao_fusao = peso_video * p_video + peso_audio * p_audio
        nivel = _nivel_de_pontuacao(pontuacao_fusao)

        alertas_criticos = []
        alertas_moderados = []

        if resultado_video:
            for alerta in resultado_video.alertas_criticos:
                if "CRITICO" in alerta.upper():
                    alertas_criticos.append(f"[VIDEO] {alerta}")
                else:
                    alertas_moderados.append(f"[VIDEO] {alerta}")

        if resultado_audio:
            for alerta in resultado_audio.alertas:
                if "CRITICO" in alerta.upper() or "ALERTA CRITICO" in alerta.upper():
                    alertas_criticos.append(f"[AUDIO] {alerta}")
                else:
                    alertas_moderados.append(f"[AUDIO] {alerta}")

        recomendacoes = _recomendacoes_por_nivel(nivel, resultado_video, resultado_audio)
        resumo = self._gerar_resumo(nivel, pontuacao_fusao, p_video, p_audio, tipo_atendimento)

        return RiscoMultimodal(
            pontuacao_video=round(p_video, 4),
            pontuacao_audio=round(p_audio, 4),
            pontuacao_fusao=round(pontuacao_fusao, 4),
            nivel_prioridade=nivel,
            alertas_criticos=alertas_criticos,
            alertas_moderados=alertas_moderados,
            recomendacoes=recomendacoes,
            resumo_executivo=resumo,
        )

    def _gerar_resumo(
        self,
        nivel: NivelPrioridade,
        pontuacao: float,
        p_video: float,
        p_audio: float,
        tipo: str,
    ) -> str:
        return (
            f"Avaliação multimodal concluída para atendimento '{tipo}'. "
            f"Pontuação de risco fusionada: {pontuacao:.2f} | "
            f"Vídeo: {p_video:.2f} | Áudio: {p_audio:.2f}. "
            f"Nível de prioridade: {nivel.value}."
        )
