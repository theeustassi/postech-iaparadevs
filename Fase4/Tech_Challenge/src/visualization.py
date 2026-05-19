"""
Módulo de visualização de resultados.

Gera gráficos para os resultados de análise de vídeo, áudio e fusão multimodal.
Os gráficos são salvos em results/graficos/.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.warning("matplotlib não disponível. Gráficos desabilitados.")


_DIRETORIO_GRAFICOS = Path("results/graficos")


def _garantir_diretorio():
    _DIRETORIO_GRAFICOS.mkdir(parents=True, exist_ok=True)


def plotar_risco_video_por_frame(
    timestamps: List[float],
    pontuacoes: List[float],
    tipo_video: str,
    nome_arquivo: str = "risco_video.png",
) -> Optional[str]:
    if not _MPL_OK or not timestamps:
        return None
    _garantir_diretorio()

    fig, ax = plt.subplots(figsize=(10, 4))
    cores = ["green" if p < 0.2 else ("orange" if p < 0.5 else "red") for p in pontuacoes]
    ax.bar(timestamps, pontuacoes, color=cores, width=0.4, alpha=0.8)
    ax.axhline(0.20, color="orange", linestyle="--", linewidth=1, label="Limiar moderado (0.20)")
    ax.axhline(0.50, color="red", linestyle="--", linewidth=1, label="Limiar alto (0.50)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Pontuação de risco")
    ax.set_title(f"Risco por frame - {tipo_video}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    caminho = _DIRETORIO_GRAFICOS / nome_arquivo
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", caminho)
    return str(caminho)


def plotar_pontuacoes_audio(
    pontuacoes: dict,
    tipo_consulta: str,
    nome_arquivo: str = "pontuacoes_audio.png",
) -> Optional[str]:
    if not _MPL_OK:
        return None
    _garantir_diretorio()

    labels = list(pontuacoes.keys())
    valores = list(pontuacoes.values())
    cores = ["red" if v > 0.5 else ("orange" if v > 0.25 else "green") for v in valores]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, valores, color=cores, alpha=0.85)
    ax.set_ylabel("Pontuação de risco (0-1)")
    ax.set_title(f"Indicadores de risco na consulta - {tipo_consulta}")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
    for bar, valor in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{valor:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    caminho = _DIRETORIO_GRAFICOS / nome_arquivo
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", caminho)
    return str(caminho)


def plotar_fusao_multimodal(
    resultados_fusao: List[dict],
    nome_arquivo: str = "fusao_multimodal.png",
) -> Optional[str]:
    """
    Plota o painel comparativo de todos os atendimentos analisados.
    resultados_fusao: lista de dicts com chaves 'tipo', 'video', 'audio', 'fusao'.
    """
    if not _MPL_OK or not resultados_fusao:
        return None
    _garantir_diretorio()

    tipos = [r["tipo"] for r in resultados_fusao]
    p_video = [r.get("video", 0) for r in resultados_fusao]
    p_audio = [r.get("audio", 0) for r in resultados_fusao]
    p_fusao = [r.get("fusao", 0) for r in resultados_fusao]

    x = np.arange(len(tipos))
    largura = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - largura, p_video, largura, label="Vídeo", color="steelblue", alpha=0.85)
    ax.bar(x, p_audio, largura, label="Áudio", color="darkorange", alpha=0.85)
    ax.bar(x + largura, p_fusao, largura, label="Fusão", color="firebrick", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tipos, rotation=15, ha="right")
    ax.set_ylabel("Pontuação de risco")
    ax.set_title("Comparativo de risco multimodal por atendimento")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.45, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    caminho = _DIRETORIO_GRAFICOS / nome_arquivo
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", caminho)
    return str(caminho)
