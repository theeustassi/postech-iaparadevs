"""
Ponto de entrada principal - Sistema de Monitoramento Multimodal para Saúde da Mulher.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
Autor: Matheus Tassi Souza - RM367424

Este script demonstra o pipeline completo de análise multimodal:
    1. Análise de vídeo clínico (cirurgia, consulta, fisioterapia, triagem)
    2. Análise de áudio de consultas médicas
    3. Fusão multimodal dos resultados
    4. Geração de relatórios clínicos automatizados

Uso:
    python src/main.py                        # pipeline completo (sintético)
    python src/main.py --tipo cirurgia        # apenas vídeo cirúrgico
    python src/main.py --tipo audio           # apenas áudio
    python src/main.py --video caminho.mp4 --audio caminho.wav --tipo cirurgia
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audio_analysis import AnalisadorAudio, TipoConsulta
from multimodal_fusion import FusorMultimodal
from report_generator import GeradorRelatorio
from security import AuditLogger, ValidadorEntrada
from video_analysis import AnalisadorVideo, TipoVideo
from visualization import plotar_fusao_multimodal, plotar_pontuacoes_audio, plotar_risco_video_por_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

_MAPA_TIPO_VIDEO = {
    "cirurgia": TipoVideo.CIRURGIA,
    "consulta": TipoVideo.CONSULTA,
    "fisioterapia": TipoVideo.FISIOTERAPIA,
    "triagem": TipoVideo.TRIAGEM_VIOLENCIA,
}

_MAPA_TIPO_AUDIO = {
    "ginecologica": TipoConsulta.GINECOLOGICA,
    "pre_natal": TipoConsulta.PRE_NATAL,
    "pos_parto": TipoConsulta.POS_PARTO,
    "triagem": TipoConsulta.TRIAGEM_VIOLENCIA,
}


def _cenarios_demo() -> list:
    """
    Define os cenários de demonstração sintética.
    Cada cenário combina um tipo de vídeo e um tipo de áudio.
    """
    return [
        {
            "paciente_id": "PAC-DEMO-001",
            "tipo_atendimento": "cirurgia",
            "tipo_video": TipoVideo.CIRURGIA,
            "tipo_audio": TipoConsulta.GINECOLOGICA,
        },
        {
            "paciente_id": "PAC-DEMO-002",
            "tipo_atendimento": "pos_parto",
            "tipo_video": TipoVideo.CONSULTA,
            "tipo_audio": TipoConsulta.POS_PARTO,
        },
        {
            "paciente_id": "PAC-DEMO-003",
            "tipo_atendimento": "triagem_violencia",
            "tipo_video": TipoVideo.TRIAGEM_VIOLENCIA,
            "tipo_audio": TipoConsulta.TRIAGEM_VIOLENCIA,
        },
        {
            "paciente_id": "PAC-DEMO-004",
            "tipo_atendimento": "pre_natal",
            "tipo_video": TipoVideo.CONSULTA,
            "tipo_audio": TipoConsulta.PRE_NATAL,
        },
    ]


def executar_pipeline_sintetico(
    analisador_video: AnalisadorVideo,
    analisador_audio: AnalisadorAudio,
    fusor: FusorMultimodal,
    gerador: GeradorRelatorio,
    audit: AuditLogger,
) -> list:
    """Executa o pipeline completo com dados sintéticos para todos os cenários."""
    cenarios = _cenarios_demo()
    resultados_fusao_grafico = []

    for cenario in cenarios:
        paciente_id = cenario["paciente_id"]
        tipo_atendimento = cenario["tipo_atendimento"]

        logger.info(
            "=== Processando cenario: %s | Paciente: %s ===",
            tipo_atendimento,
            paciente_id,
        )

        # 1. Analise de video
        logger.info("Etapa 1/3 - Análise de vídeo (%s)...", cenario["tipo_video"].value)
        resultado_video = analisador_video.processar_frames_sinteticos(
            cenario["tipo_video"], n_frames=30
        )
        audit.registrar_analise_video(
            paciente_id,
            resultado_video.tipo.value,
            resultado_video.pontuacao_risco_media,
            len(resultado_video.frames_com_alerta),
        )

        # 2. Analise de audio
        logger.info("Etapa 2/3 - Análise de áudio (%s)...", cenario["tipo_audio"].value)
        resultado_audio = analisador_audio.processar_sintetico(cenario["tipo_audio"])
        audit.registrar_analise_audio(
            paciente_id,
            resultado_audio.tipo_consulta.value,
            resultado_audio.pontuacao_risco_geral,
            len(resultado_audio.alertas),
        )

        # 3. Fusao multimodal
        logger.info("Etapa 3/3 - Fusão multimodal...")
        risco = fusor.fundir(resultado_video, resultado_audio, tipo_atendimento)
        audit.registrar_fusao(
            paciente_id,
            tipo_atendimento,
            risco.nivel_prioridade.value,
            risco.pontuacao_fusao,
        )

        for alerta in risco.alertas_criticos:
            audit.registrar_alerta_critico(paciente_id, alerta)

        # Gerar e imprimir relatorio
        relatorio = gerador.gerar(
            paciente_id=paciente_id,
            tipo_atendimento=tipo_atendimento,
            resultado_video=resultado_video,
            resultado_audio=resultado_audio,
            risco_multimodal=risco,
            salvar=True,
        )
        gerador.imprimir_relatorio(relatorio)

        # Graficos individuais
        timestamps = [f.timestamp_s for f in resultado_video.frames_com_alerta]
        pontuacoes = [f.pontuacao_risco for f in resultado_video.frames_com_alerta]
        if not timestamps:
            timestamps = [f.timestamp_s for f in resultado_video.frames_processados if hasattr(resultado_video, 'frames_processados')] or list(range(resultado_video.frames_processados))
            pontuacoes = [0.0] * len(timestamps)

        # Usa todos os frames para o grafico de video
        if resultado_video.frames_com_alerta:
            plotar_risco_video_por_frame(
                [f.timestamp_s for f in resultado_video.frames_com_alerta],
                [f.pontuacao_risco for f in resultado_video.frames_com_alerta],
                resultado_video.tipo.value,
                f"risco_video_{tipo_atendimento}.png",
            )

        plotar_pontuacoes_audio(
            {
                "Depressão": resultado_audio.pontuacao_depressao,
                "Ansiedade": resultado_audio.pontuacao_ansiedade,
                "Violência": resultado_audio.pontuacao_violencia,
                "Fadiga": resultado_audio.pontuacao_fadiga,
            },
            resultado_audio.tipo_consulta.value,
            f"pontuacoes_audio_{tipo_atendimento}.png",
        )

        resultados_fusao_grafico.append({
            "tipo": tipo_atendimento,
            "video": risco.pontuacao_video,
            "audio": risco.pontuacao_audio,
            "fusao": risco.pontuacao_fusao,
        })

    plotar_fusao_multimodal(resultados_fusao_grafico, "fusao_multimodal_comparativo.png")
    return resultados_fusao_grafico


def executar_pipeline_arquivo(
    caminho_video: str,
    caminho_audio: str,
    tipo_video_str: str,
    tipo_audio_str: str,
    paciente_id: str,
    analisador_video: AnalisadorVideo,
    analisador_audio: AnalisadorAudio,
    fusor: FusorMultimodal,
    gerador: GeradorRelatorio,
    audit: AuditLogger,
) -> None:
    """Executa o pipeline com arquivos reais fornecidos pelo usuário."""
    validador = ValidadorEntrada()

    if not validador.validar_id_paciente(paciente_id):
        logger.error("ID de paciente inválido: %s", paciente_id)
        sys.exit(1)

    tipo_video = _MAPA_TIPO_VIDEO.get(tipo_video_str.lower())
    tipo_audio = _MAPA_TIPO_AUDIO.get(tipo_audio_str.lower())

    if tipo_video is None:
        logger.error("Tipo de vídeo inválido: %s. Opções: %s", tipo_video_str, list(_MAPA_TIPO_VIDEO.keys()))
        sys.exit(1)
    if tipo_audio is None:
        logger.error("Tipo de áudio inválido: %s. Opções: %s", tipo_audio_str, list(_MAPA_TIPO_AUDIO.keys()))
        sys.exit(1)

    resultado_video = None
    resultado_audio = None

    if caminho_video:
        logger.info("Analisando video: %s", caminho_video)
        resultado_video = analisador_video.processar_arquivo(caminho_video, tipo_video)
        audit.registrar_analise_video(
            paciente_id,
            resultado_video.tipo.value,
            resultado_video.pontuacao_risco_media,
            len(resultado_video.frames_com_alerta),
        )

    if caminho_audio:
        logger.info("Analisando audio: %s", caminho_audio)
        resultado_audio = analisador_audio.processar_arquivo(caminho_audio, tipo_audio)
        audit.registrar_analise_audio(
            paciente_id,
            resultado_audio.tipo_consulta.value,
            resultado_audio.pontuacao_risco_geral,
            len(resultado_audio.alertas),
        )

    tipo_atendimento = tipo_video_str.lower()
    risco = fusor.fundir(resultado_video, resultado_audio, tipo_atendimento)
    audit.registrar_fusao(paciente_id, tipo_atendimento, risco.nivel_prioridade.value, risco.pontuacao_fusao)

    relatorio = gerador.gerar(
        paciente_id=paciente_id,
        tipo_atendimento=tipo_atendimento,
        resultado_video=resultado_video,
        resultado_audio=resultado_audio,
        risco_multimodal=risco,
        salvar=True,
    )
    gerador.imprimir_relatorio(relatorio)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Monitoramento Multimodal para Saúde da Mulher - HospitalIQ"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Caminho para arquivo de video a analisar",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Caminho para arquivo de audio (.wav) a analisar",
    )
    parser.add_argument(
        "--tipo",
        type=str,
        default="cirurgia",
        choices=list(_MAPA_TIPO_VIDEO.keys()),
        help="Tipo de vídeo/atendimento",
    )
    parser.add_argument(
        "--tipo-audio",
        type=str,
        default=None,
        choices=list(_MAPA_TIPO_AUDIO.keys()),
        help="Tipo de consulta para o áudio (padrão: mesmo que --tipo quando aplicável)",
    )
    parser.add_argument(
        "--paciente-id",
        type=str,
        default="PAC-DEMO-001",
        help="Identificador da paciente (será anonimizado nos logs)",
    )
    args = parser.parse_args()

    analisador_video = AnalisadorVideo(fps_processamento=2, max_frames=300)
    analisador_audio = AnalisadorAudio()
    fusor = FusorMultimodal()
    gerador = GeradorRelatorio()
    audit = AuditLogger()

    if args.video is None and args.audio is None:
        logger.info("Nenhum arquivo fornecido. Executando demonstração com dados sintéticos.")
        executar_pipeline_sintetico(analisador_video, analisador_audio, fusor, gerador, audit)
    else:
        tipo_audio = args.tipo_audio or args.tipo
        executar_pipeline_arquivo(
            caminho_video=args.video or "",
            caminho_audio=args.audio or "",
            tipo_video_str=args.tipo,
            tipo_audio_str=tipo_audio,
            paciente_id=args.paciente_id,
            analisador_video=analisador_video,
            analisador_audio=analisador_audio,
            fusor=fusor,
            gerador=gerador,
            audit=audit,
        )

    logger.info("Pipeline concluído. Resultados em results/")


if __name__ == "__main__":
    main()
