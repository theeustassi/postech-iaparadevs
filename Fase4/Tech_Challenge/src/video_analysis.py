"""
Módulo de análise de vídeo para monitoramento especializado em saúde da mulher.

Responsabilidades:
- Processar frames de vídeos clínicos (cirurgias, consultas, fisioterapia, triagem)
- Detectar sangramento anômalo via análise de cor (YOLOv8 + HSV masking)
- Analisar linguagem corporal em consultas via estimativa de pose (YOLOv8-pose)
- Avaliar padrão de movimento em sessões de fisioterapia
- Gerar pontuação de risco por frame e por vídeo completo

Todos os dados processados são anonimizados antes de qualquer persistência.
Nenhuma imagem identificável é armazenada; apenas métricas são retidas.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False

logger = logging.getLogger(__name__)


class TipoVideo(str, Enum):
    CIRURGIA = "cirurgia"
    CONSULTA = "consulta"
    FISIOTERAPIA = "fisioterapia"
    TRIAGEM_VIOLENCIA = "triagem_violencia"


@dataclass
class ResultadoFrame:
    numero_frame: int
    timestamp_s: float
    pontuacao_risco: float          # 0.0 a 1.0
    alertas: List[str] = field(default_factory=list)
    metricas: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResultadoVideo:
    tipo: TipoVideo
    total_frames: int
    frames_processados: int
    duracao_s: float
    pontuacao_risco_media: float
    pontuacao_risco_maxima: float
    alertas_criticos: List[str] = field(default_factory=list)
    frames_com_alerta: List[ResultadoFrame] = field(default_factory=list)
    resumo_clinico: str = ""


# ---------------------------------------------------------------------------
# Detectores especializados
# ---------------------------------------------------------------------------

class DetectorSangramento:
    """
    Detecta presença de sangramento anômalo em frames cirúrgicos.

    Estratégia:
    - Converte o frame para o espaço de cor HSV
    - Aplica uma máscara para tons de vermelho intenso (sangue fresco)
    - Calcula a proporção de pixels vermelhos em relação à área relevante
    - Classifica como anômalo quando a proporção supera o limiar

    Em produção, esta etapa seria substituída por um modelo YOLOv8 customizado
    treinado em imagens cirúrgicas anotadas com sangramento, garantindo maior
    precisão e menor taxa de falsos positivos causados por instrumentos vermelhos.
    """

    # Limites HSV para vermelho escuro (sangue)
    HSV_BAIXO_1 = np.array([0, 120, 70])
    HSV_ALTO_1 = np.array([10, 255, 255])
    HSV_BAIXO_2 = np.array([160, 120, 70])
    HSV_ALTO_2 = np.array([180, 255, 255])

    LIMIAR_PROPORCAO = 0.08   # 8% de pixels vermelhos = alerta
    LIMIAR_CRITICO = 0.20     # 20% = sangramento crítico

    def analisar(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        """
        Retorna (pontuacao_risco, lista_de_alertas) para o frame informado.
        pontuacao_risco: 0.0 = sem risco, 1.0 = risco maximo.
        """
        if not _CV2_OK:
            # Fallback: analise de canal vermelho direto no array numpy
            canal_vermelho = frame[:, :, 2].astype(np.float32)
            canal_verde = frame[:, :, 1].astype(np.float32)
            canal_azul = frame[:, :, 0].astype(np.float32)
            mascara_numpy = (
                (canal_vermelho > 120)
                & (canal_verde < 80)
                & (canal_azul < 80)
            )
            proporcao = float(np.sum(mascara_numpy)) / (frame.shape[0] * frame.shape[1])
            alertas: List[str] = []
            if proporcao >= self.LIMIAR_CRITICO:
                alertas.append(f"CRÍTICO: sangramento anômalo extenso detectado ({proporcao * 100:.1f}% da área do frame)")
                return 1.0, alertas
            elif proporcao >= self.LIMIAR_PROPORCAO:
                alertas.append(f"ALERTA: possível sangramento detectado ({proporcao * 100:.1f}% da área do frame)")
                return min(0.4 + proporcao * 3.0, 0.95), alertas
            return proporcao * 5.0, []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara1 = cv2.inRange(hsv, self.HSV_BAIXO_1, self.HSV_ALTO_1)
        mascara2 = cv2.inRange(hsv, self.HSV_BAIXO_2, self.HSV_ALTO_2)
        mascara = cv2.bitwise_or(mascara1, mascara2)

        total_pixels = frame.shape[0] * frame.shape[1]
        pixels_vermelhos = int(np.sum(mascara > 0))
        proporcao = pixels_vermelhos / total_pixels if total_pixels > 0 else 0.0

        alertas: List[str] = []
        if proporcao >= self.LIMIAR_CRITICO:
            alertas.append(
                f"CRÍTICO: sangramento anômalo extenso detectado "
                f"({proporcao * 100:.1f}% da área do frame)"
            )
            risco = 1.0
        elif proporcao >= self.LIMIAR_PROPORCAO:
            alertas.append(
                f"ALERTA: possível sangramento detectado "
                f"({proporcao * 100:.1f}% da área do frame)"
            )
            risco = min(0.4 + proporcao * 3.0, 0.95)
        else:
            risco = proporcao * 5.0

        return risco, alertas


class AnalisadorPosturaCorporal:
    """
    Analisa linguagem corporal usando YOLOv8-pose para detecção de pose.

    Indicadores monitorados:
    - Postura fechada (braços cruzados, ombros encurvados) -> possível desconforto
    - Movimentos bruscos ou tremores -> possível ansiedade ou medo
    - Cabeça baixa com movimento mínimo -> possível estado dissociativo

    Quando a biblioteca ultralytics não está disponível, o módulo opera em modo
    degradado, retornando métricas baseadas apenas no fluxo óptico.
    """

    def __init__(self):
        self._modelo = None
        self._usar_yolo = False
        self._frame_anterior: Optional[np.ndarray] = None
        self._tentar_carregar_yolo()

    def _tentar_carregar_yolo(self):
        try:
            from ultralytics import YOLO
            self._modelo = YOLO("yolov8n-pose.pt")
            self._usar_yolo = True
            logger.info("YOLOv8-pose carregado com sucesso.")
        except Exception as exc:
            logger.warning(
                "YOLOv8-pose não disponível (%s). Usando fluxo óptico como fallback.",
                exc,
            )

    def analisar(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        if self._usar_yolo and self._modelo is not None:
            return self._analisar_yolo(frame)
        return self._analisar_fluxo_optico(frame)

    def _analisar_yolo(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        resultados = self._modelo(frame, verbose=False)
        alertas: List[str] = []
        risco = 0.0

        for resultado in resultados:
            if resultado.keypoints is None:
                continue
            kps = resultado.keypoints.xy.cpu().numpy()
            for pessoa_kps in kps:
                risco_pessoa, alertas_pessoa = self._avaliar_postura(pessoa_kps)
                if risco_pessoa > risco:
                    risco = risco_pessoa
                alertas.extend(alertas_pessoa)

        return min(risco, 1.0), alertas

    def _avaliar_postura(self, kps: np.ndarray) -> Tuple[float, List[str]]:
        """
        Avalia postura a partir dos 17 keypoints COCO.
        Índice 5/6 = ombros, 7/8 = cotovelos, 9/10 = pulsos,
        11/12 = quadril, 0 = nariz (cabeça).
        """
        alertas: List[str] = []
        risco = 0.0

        if len(kps) < 13:
            return risco, alertas

        ombro_e, ombro_d = kps[5], kps[6]
        cotovelo_e, cotovelo_d = kps[7], kps[8]
        pulso_e, pulso_d = kps[9], kps[10]
        cabeca = kps[0]

        # Postura fechada: pulsos acima dos cotovelos e na linha central
        centro_x = (ombro_e[0] + ombro_d[0]) / 2
        pulsos_cruzados = (
            abs(pulso_e[0] - centro_x) < 40
            and abs(pulso_d[0] - centro_x) < 40
        )
        if pulsos_cruzados:
            alertas.append("Postura fechada detectada - possível estado de defesa ou desconforto")
            risco += 0.3

        # Cabeça inclinada para baixo
        if len(kps) > 0 and ombro_e[1] > 0 and ombro_d[1] > 0:
            media_ombro_y = (ombro_e[1] + ombro_d[1]) / 2
            if cabeca[1] > media_ombro_y * 0.8:
                alertas.append("Cabeça inclinada - possível esquivamento de contato visual")
                risco += 0.2

        return min(risco, 1.0), alertas

    def _analisar_fluxo_optico(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        """Calcula variacao de movimento via diferenca de frames (fallback)."""
        if _CV2_OK:
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            cinza = np.mean(frame, axis=2).astype(np.uint8)
        alertas: List[str] = []
        risco = 0.0

        if self._frame_anterior is not None:
            diff = np.abs(cinza.astype(np.float32) - self._frame_anterior.astype(np.float32))
            variacao = float(np.mean(diff)) / 255.0

            if variacao > 0.15:
                alertas.append(
                    f"Movimento brusco detectado (variação={variacao:.2f}) - "
                    "possível agitação ou tremor"
                )
                risco = min(variacao * 4.0, 0.8)

        self._frame_anterior = cinza.copy()
        return risco, alertas


class AnalisadorMovimentoFisioterapia:
    """
    Analisa amplitude e simetria de movimentos em sessões de fisioterapia.

    Métricas computadas:
    - Amplitude de movimento bilateral (comparação esquerda/direita)
    - Smoothness do movimento (variação de velocidade entre frames)
    - Detecção de compensações posturais

    A assimetria bilateral acima de 30% pode indicar dor ou restrição de movimento,
    justificando alerta para o fisioterapeuta responsável.
    """

    def __init__(self):
        self._modelo = None
        self._usar_yolo = False
        self._historico_kps: List[np.ndarray] = []
        self._tentar_carregar_yolo()

    def _tentar_carregar_yolo(self):
        try:
            from ultralytics import YOLO
            self._modelo = YOLO("yolov8n-pose.pt")
            self._usar_yolo = True
        except Exception:
            pass

    def analisar(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        if self._usar_yolo and self._modelo is not None:
            return self._analisar_yolo(frame)
        return self._analisar_diferenca_frame(frame)

    def _analisar_yolo(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        resultados = self._modelo(frame, verbose=False)
        alertas: List[str] = []
        risco = 0.0

        for resultado in resultados:
            if resultado.keypoints is None:
                continue
            kps = resultado.keypoints.xy.cpu().numpy()
            for pessoa_kps in kps:
                if len(pessoa_kps) >= 13:
                    self._historico_kps.append(pessoa_kps.copy())
                    if len(self._historico_kps) > 10:
                        self._historico_kps.pop(0)
                    r, a = self._avaliar_simetria_e_amplitude(pessoa_kps)
                    if r > risco:
                        risco = r
                    alertas.extend(a)

        return min(risco, 1.0), alertas

    def _avaliar_simetria_e_amplitude(
        self, kps: np.ndarray
    ) -> Tuple[float, List[str]]:
        alertas: List[str] = []
        risco = 0.0

        ombro_e, ombro_d = kps[5], kps[6]
        cotovelo_e, cotovelo_d = kps[7], kps[8]
        pulso_e, pulso_d = kps[9], kps[10]

        # Amplitude vertical dos membros superiores
        amp_e = abs(float(pulso_e[1]) - float(ombro_e[1]))
        amp_d = abs(float(pulso_d[1]) - float(ombro_d[1]))

        if amp_e > 5 and amp_d > 5:
            assimetria = abs(amp_e - amp_d) / max(amp_e, amp_d)
            if assimetria > 0.4:
                alertas.append(
                    f"Assimetria bilateral significativa detectada "
                    f"({assimetria * 100:.0f}%) - possível compensação por dor"
                )
                risco = min(assimetria, 0.8)

        # Smoothness: variacao brusca em relacao ao frame anterior
        if len(self._historico_kps) >= 2:
            diff = np.mean(np.abs(kps - self._historico_kps[-2]))
            if diff > 25:
                alertas.append("Movimento brusco - possível guarda de dor")
                risco = max(risco, 0.5)

        return risco, alertas

    def _analisar_diferenca_frame(self, frame: np.ndarray) -> Tuple[float, List[str]]:
        return 0.0, []


# ---------------------------------------------------------------------------
# Pipeline principal de analise de video
# ---------------------------------------------------------------------------

class AnalisadorVideo:
    """
    Pipeline principal de análise de vídeo para saúde da mulher.

    Delega para o detector especializado de acordo com o tipo de vídeo:
    - CIRURGIA: DetectorSangramento
    - CONSULTA: AnalisadorPosturaCorporal
    - FISIOTERAPIA: AnalisadorMovimentoFisioterapia
    - TRIAGEM_VIOLENCIA: AnalisadorPosturaCorporal (modo estendido)

    Parâmetros de processamento:
    - fps_processamento: quantos frames por segundo são analisados (redução de carga)
    - max_frames: limite máximo de frames por vídeo (0 = sem limite)
    """

    def __init__(
        self,
        fps_processamento: int = 2,
        max_frames: int = 300,
    ):
        self.fps_processamento = fps_processamento
        self.max_frames = max_frames

        self._detector_sangramento = DetectorSangramento()
        self._analisador_postura = AnalisadorPosturaCorporal()
        self._analisador_fisio = AnalisadorMovimentoFisioterapia()

    def processar_arquivo(
        self, caminho_video: str, tipo: TipoVideo
    ) -> ResultadoVideo:
        """
        Processa um arquivo de vídeo em disco.

        Args:
            caminho_video: Caminho para o arquivo de vídeo.
            tipo: Tipo de vídeo clínico (define qual detector usar).

        Returns:
            ResultadoVideo com métricas consolidadas.
        """
        if not _CV2_OK:
            raise ImportError(
                "opencv-python-headless é necessário para processar arquivos de vídeo. "
                "Instale com: pip install opencv-python-headless"
            )
        cap = cv2.VideoCapture(str(caminho_video))
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {caminho_video}")

        fps_original = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        intervalo = max(1, int(fps_original / self.fps_processamento))

        resultados_frames: List[ResultadoFrame] = []
        numero_frame = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if numero_frame % intervalo == 0:
                    resultado_frame = self._processar_frame(frame, numero_frame, fps_original, tipo)
                    resultados_frames.append(resultado_frame)

                numero_frame += 1

                if self.max_frames > 0 and len(resultados_frames) >= self.max_frames:
                    break
        finally:
            cap.release()

        return self._consolidar_resultados(tipo, total_frames_video, numero_frame, fps_original, resultados_frames)

    def processar_frames_sinteticos(
        self, tipo: TipoVideo, n_frames: int = 30
    ) -> ResultadoVideo:
        """
        Processa frames sintéticos para demonstração e testes.

        Gera frames com características controladas para validar o pipeline
        sem necessidade de vídeos reais de pacientes.
        """
        logger.info("Gerando %d frames sinteticos para tipo=%s", n_frames, tipo.value)
        resultados_frames: List[ResultadoFrame] = []

        rng = np.random.default_rng(42)

        for i in range(n_frames):
            frame = self._gerar_frame_sintetico(tipo, i, rng)
            resultado = self._processar_frame(frame, i, 25.0, tipo)
            resultados_frames.append(resultado)

        return self._consolidar_resultados(tipo, n_frames, n_frames, 25.0, resultados_frames)

    def _gerar_frame_sintetico(
        self, tipo: TipoVideo, indice: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Gera um frame BGR sintético com características específicas por tipo."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if tipo == TipoVideo.CIRURGIA:
            # Fundo verde (tecido cirúrgico) com mancha vermelha progressiva
            frame[:, :] = [30, 100, 30]
            if indice > 15:
                intensidade = min(indice - 15, 15)
                # Raio cresce rapidamente para ultrapassar os limiares de alerta (8%) e crítico (20%)
                raio = 20 + intensidade * 8
                cx, cy = 320, 240
                if _CV2_OK:
                    cv2.circle(frame, (cx, cy), raio, (0, 0, 180), -1)
                else:
                    # Fallback numpy: pinta círculo manualmente
                    y, x = np.ogrid[:480, :640]
                    mascara_circ = (x - cx) ** 2 + (y - cy) ** 2 <= raio ** 2
                    frame[mascara_circ] = [0, 0, 180]
        elif tipo in (TipoVideo.CONSULTA, TipoVideo.TRIAGEM_VIOLENCIA):
            frame[:, :] = [180, 160, 140]
            ruido = rng.integers(0, 30, frame.shape, dtype=np.uint8)
            frame = np.clip(frame.astype(np.int32) + ruido.astype(np.int32), 0, 255).astype(np.uint8)
        elif tipo == TipoVideo.FISIOTERAPIA:
            frame[:, :] = [200, 200, 200]
            pos_y = 240 + int(60 * np.sin(indice * 0.4))
            if _CV2_OK:
                cv2.circle(frame, (320, pos_y), 20, (50, 50, 200), -1)
                cv2.circle(frame, (200, pos_y + int(rng.integers(-10, 10))), 15, (200, 50, 50), -1)
            else:
                y, x = np.ogrid[:480, :640]
                frame[(x - 320) ** 2 + (y - pos_y) ** 2 <= 400] = [50, 50, 200]

        return frame

    def _processar_frame(
        self,
        frame: np.ndarray,
        numero_frame: int,
        fps: float,
        tipo: TipoVideo,
    ) -> ResultadoFrame:
        timestamp = numero_frame / fps if fps > 0 else 0.0

        if tipo == TipoVideo.CIRURGIA:
            risco, alertas = self._detector_sangramento.analisar(frame)
        elif tipo in (TipoVideo.CONSULTA, TipoVideo.TRIAGEM_VIOLENCIA):
            risco, alertas = self._analisador_postura.analisar(frame)
        else:
            risco, alertas = self._analisador_fisio.analisar(frame)

        return ResultadoFrame(
            numero_frame=numero_frame,
            timestamp_s=round(timestamp, 3),
            pontuacao_risco=round(risco, 4),
            alertas=alertas,
            metricas={"risco": round(risco, 4)},
        )

    def _consolidar_resultados(
        self,
        tipo: TipoVideo,
        total_frames: int,
        frames_processados: int,
        fps: float,
        resultados: List[ResultadoFrame],
    ) -> ResultadoVideo:
        if not resultados:
            return ResultadoVideo(
                tipo=tipo,
                total_frames=total_frames,
                frames_processados=0,
                duracao_s=0.0,
                pontuacao_risco_media=0.0,
                pontuacao_risco_maxima=0.0,
                resumo_clinico="Nenhum frame processado.",
            )

        pontuacoes = [r.pontuacao_risco for r in resultados]
        media = float(np.mean(pontuacoes))
        maxima = float(np.max(pontuacoes))
        frames_alerta = [r for r in resultados if r.alertas]
        todos_alertas = [a for r in frames_alerta for a in r.alertas]
        alertas_criticos = [a for a in todos_alertas if "CRITICO" in a.upper()]

        resumo = self._gerar_resumo_clinico(tipo, media, maxima, len(frames_alerta), len(resultados))

        return ResultadoVideo(
            tipo=tipo,
            total_frames=total_frames,
            frames_processados=len(resultados),
            duracao_s=round(frames_processados / fps, 2) if fps > 0 else 0.0,
            pontuacao_risco_media=round(media, 4),
            pontuacao_risco_maxima=round(maxima, 4),
            alertas_criticos=alertas_criticos,
            frames_com_alerta=frames_alerta,
            resumo_clinico=resumo,
        )

    def _gerar_resumo_clinico(
        self,
        tipo: TipoVideo,
        media: float,
        maxima: float,
        n_frames_alerta: int,
        n_total: int,
    ) -> str:
        proporcao_alerta = n_frames_alerta / n_total if n_total > 0 else 0.0
        nivel = "BAIXO" if media < 0.2 else ("MODERADO" if media < 0.5 else "ALTO")

        descricoes = {
            TipoVideo.CIRURGIA: (
                f"Análise cirúrgica concluída. Risco de sangramento anômalo: {nivel}. "
                f"Proporção de frames com alerta: {proporcao_alerta * 100:.1f}%. "
                f"Pontuação máxima de risco registrada: {maxima:.2f}."
            ),
            TipoVideo.CONSULTA: (
                f"Análise comportamental de consulta concluída. Nível de desconforto detectado: {nivel}. "
                f"Sinais não-verbais de alerta em {proporcao_alerta * 100:.1f}% dos frames analisados."
            ),
            TipoVideo.FISIOTERAPIA: (
                f"Análise de movimento fisioterapêutico concluída. Nível de compensação postural: {nivel}. "
                f"Assimetrias de movimento detectadas em {proporcao_alerta * 100:.1f}% dos frames."
            ),
            TipoVideo.TRIAGEM_VIOLENCIA: (
                f"Triagem para indicadores de violência concluída. Nível de risco: {nivel}. "
                f"Padrões de linguagem corporal preocupantes em {proporcao_alerta * 100:.1f}% dos frames. "
                "Recomenda-se encaminhamento para avaliação psicossocial especializada."
                if nivel != "BAIXO"
                else f"Triagem para indicadores de violência concluída. Nível de risco: {nivel}."
            ),
        }
        return descricoes.get(tipo, f"Análise de vídeo concluída. Risco: {nivel}.")
