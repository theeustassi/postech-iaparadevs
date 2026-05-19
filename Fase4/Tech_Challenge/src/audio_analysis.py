"""
Módulo de análise de áudio para monitoramento especializado em saúde da mulher.

Responsabilidades:
- Extrair features acústicas relevantes (MFCC, pitch, energia, taxa de pausa)
- Transcrever fala via OpenAI Whisper (quando disponível)
- Calcular pontuações de risco para:
    - Depressão pós-parto (prosódia plana, fala lenta, padrões de pausas longas)
    - Ansiedade gestacional (fala acelerada, variação de pitch elevada)
    - Indicadores de violência doméstica (hesitação, voz baixa, tremores vocais)
    - Fadiga hormonal (energia vocal reduzida, monotonia)
- Gerar laudo textual com recomendações clínicas

Aviso: o módulo opera de forma totalmente local. Nenhum dado de áudio é transmitido
para serviços externos. Todos os resultados são anonimizados antes da persistência.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
"""

import logging
import re
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tentativa de importar dependencias opcionais
try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("librosa não disponível. Features acústicas serão calculadas via numpy.")

try:
    import whisper as openai_whisper
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    logger.warning("openai-whisper não disponível. Transcrição desabilitada.")

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False
    logger.warning("transformers não disponível. Análise de sentimento baseada em regras.")


class TipoConsulta(str, Enum):
    GINECOLOGICA = "ginecologica"
    PRE_NATAL = "pre_natal"
    POS_PARTO = "pos_parto"
    TRIAGEM_VIOLENCIA = "triagem_violencia"


@dataclass
class FeaturesAcusticas:
    duracao_s: float
    energia_media: float           # 0.0 - 1.0 (normalizado)
    pitch_medio_hz: float
    pitch_desvio_hz: float
    taxa_pausa: float              # proporcao do audio que e silencio
    pausa_maxima_s: float
    taxa_fala_silabas_s: float     # estimativa de silabas por segundo
    mfcc_media: List[float] = field(default_factory=list)
    mfcc_desvio: List[float] = field(default_factory=list)


@dataclass
class ResultadoAudio:
    tipo_consulta: TipoConsulta
    duracao_s: float
    features: FeaturesAcusticas
    transcricao: str
    pontuacao_depressao: float     # 0.0 - 1.0
    pontuacao_ansiedade: float     # 0.0 - 1.0
    pontuacao_violencia: float     # 0.0 - 1.0
    pontuacao_fadiga: float        # 0.0 - 1.0
    pontuacao_risco_geral: float   # 0.0 - 1.0
    alertas: List[str] = field(default_factory=list)
    resumo_clinico: str = ""


# ---------------------------------------------------------------------------
# Extratores de features acusticas
# ---------------------------------------------------------------------------

class ExtratorFeatures:
    """
    Extrai features acústicas relevantes para análise de saúde vocal.

    Usa librosa quando disponível (alta precisão). Quando não disponível,
    usa implementação numpy (acurácia reduzida, suficiente para demo).
    """

    TAXA_AMOSTRAGEM_PADRAO = 16000
    HOP_LENGTH = 512
    N_MFCC = 13

    def extrair_de_arquivo(self, caminho: str) -> Tuple[FeaturesAcusticas, np.ndarray, int]:
        """
        Retorna (features, sinal_audio, taxa_amostragem).
        """
        if _LIBROSA_OK:
            return self._extrair_librosa(caminho)
        return self._extrair_scipy(caminho)

    def extrair_de_array(
        self, sinal: np.ndarray, sr: int
    ) -> FeaturesAcusticas:
        """
        Extrai features diretamente de um array numpy.
        Conveniente para testes com áudio sintético.
        """
        if _LIBROSA_OK:
            return self._features_librosa(sinal, sr)
        return self._features_numpy(sinal, sr)

    # ------------------------------------------------------------------

    def _extrair_librosa(
        self, caminho: str
    ) -> Tuple[FeaturesAcusticas, np.ndarray, int]:
        sinal, sr = librosa.load(caminho, sr=self.TAXA_AMOSTRAGEM_PADRAO, mono=True)
        return self._features_librosa(sinal, sr), sinal, sr

    def _features_librosa(self, sinal: np.ndarray, sr: int) -> FeaturesAcusticas:
        duracao = len(sinal) / sr

        # Energia RMS
        rms = librosa.feature.rms(y=sinal, hop_length=self.HOP_LENGTH)[0]
        energia_media = float(np.mean(rms))
        energia_max = float(np.max(rms)) if np.max(rms) > 0 else 1.0
        energia_normalizada = energia_media / energia_max

        # Pitch (F0)
        f0, voiced_flag, _ = librosa.pyin(
            sinal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        pitch_medio = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_desvio = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0

        # Pausas (frames com energia abaixo de 10% da media)
        limiar_silencio = energia_media * 0.1
        frames_silencio = np.sum(rms < limiar_silencio)
        taxa_pausa = float(frames_silencio) / len(rms) if len(rms) > 0 else 0.0
        pausa_maxima = self._pausa_maxima(rms, limiar_silencio, sr)

        # Taxa de fala (estimada pelo numero de onsets)
        onsets = librosa.onset.onset_detect(y=sinal, sr=sr, units="time")
        taxa_fala = float(len(onsets)) / duracao if duracao > 0 else 0.0

        # MFCC
        mfcc = librosa.feature.mfcc(y=sinal, sr=sr, n_mfcc=self.N_MFCC)
        mfcc_media = np.mean(mfcc, axis=1).tolist()
        mfcc_desvio = np.std(mfcc, axis=1).tolist()

        return FeaturesAcusticas(
            duracao_s=round(duracao, 2),
            energia_media=round(float(energia_normalizada), 4),
            pitch_medio_hz=round(pitch_medio, 2),
            pitch_desvio_hz=round(pitch_desvio, 2),
            taxa_pausa=round(taxa_pausa, 4),
            pausa_maxima_s=round(pausa_maxima, 3),
            taxa_fala_silabas_s=round(taxa_fala, 3),
            mfcc_media=[round(v, 3) for v in mfcc_media],
            mfcc_desvio=[round(v, 3) for v in mfcc_desvio],
        )

    def _pausa_maxima(
        self, rms: np.ndarray, limiar: float, sr: int
    ) -> float:
        frames_por_segundo = sr / self.HOP_LENGTH
        max_consecutivo = 0
        atual = 0
        for v in rms:
            if v < limiar:
                atual += 1
                max_consecutivo = max(max_consecutivo, atual)
            else:
                atual = 0
        return float(max_consecutivo) / frames_por_segundo

    def _extrair_scipy(
        self, caminho: str
    ) -> Tuple[FeaturesAcusticas, np.ndarray, int]:
        """Fallback usando apenas biblioteca padrao + numpy."""
        try:
            with wave.open(caminho, "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(n_frames)
        except Exception as exc:
            raise ValueError(f"Não foi possível abrir áudio: {caminho}") from exc

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        sinal = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        if n_channels > 1:
            sinal = sinal.reshape(-1, n_channels).mean(axis=1)

        sinal /= np.iinfo(dtype).max if np.iinfo(dtype).max > 0 else 1.0
        return self._features_numpy(sinal, sr), sinal, sr

    def _features_numpy(self, sinal: np.ndarray, sr: int) -> FeaturesAcusticas:
        duracao = len(sinal) / sr
        energia_rms = float(np.sqrt(np.mean(sinal ** 2)))
        taxa_pausa = float(np.mean(np.abs(sinal) < 0.01))
        return FeaturesAcusticas(
            duracao_s=round(duracao, 2),
            energia_media=round(energia_rms, 4),
            pitch_medio_hz=0.0,
            pitch_desvio_hz=0.0,
            taxa_pausa=round(taxa_pausa, 4),
            pausa_maxima_s=0.0,
            taxa_fala_silabas_s=0.0,
        )


# ---------------------------------------------------------------------------
# Transcricao com Whisper
# ---------------------------------------------------------------------------

class TranscritorAudio:
    """
    Transcreve áudio usando OpenAI Whisper (modelo small, executado localmente).

    Quando Whisper não está disponível, retorna string vazia e registra aviso.
    O modelo é carregado na primeira chamada e reutilizado nas subsequentes.
    """

    _modelo = None

    def transcrever(self, sinal: np.ndarray, sr: int) -> str:
        if not _WHISPER_OK:
            return ""
        if self._modelo is None:
            logger.info("Carregando modelo Whisper (small)...")
            self._modelo = openai_whisper.load_model("small")
        try:
            if sr != 16000:
                import librosa
                sinal = librosa.resample(sinal, orig_sr=sr, target_sr=16000)
            resultado = self._modelo.transcribe(sinal.astype(np.float32), language="pt")
            return resultado.get("text", "").strip()
        except Exception as exc:
            logger.warning("Erro na transcrição: %s", exc)
            return ""

    def transcrever_arquivo(self, caminho: str) -> str:
        if not _WHISPER_OK:
            return ""
        if self._modelo is None:
            logger.info("Carregando modelo Whisper (small)...")
            self._modelo = openai_whisper.load_model("small")
        try:
            resultado = self._modelo.transcribe(caminho, language="pt")
            return resultado.get("text", "").strip()
        except Exception as exc:
            logger.warning("Erro na transcrição: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Classificadores de risco
# ---------------------------------------------------------------------------

class ClassificadorRiscoDepressao:
    """
    Estima probabilidade de depressão pós-parto a partir de features acústicas
    e texto transcrito.

    Indicadores acústicos de depressão (evidência clínica):
    - Energia vocal reduzida (hipofonia)
    - Pitch médio baixo e pouca variação (prosódia monotona)
    - Alta taxa de pausas e pausas longas (retardo psicomotor)
    - Fala lenta (taxa de sílabas/s baixa)
    """

    _PALAVRAS_DEPRESSAO = [
        "cansada", "cansaco", "sem energia", "nao consigo", "nao sei",
        "choro", "chorando", "tristeza", "triste", "sem vontade",
        "nao durmo", "nao estou dormindo", "nao me sinto bem",
        "nao gosto", "arrependida", "culpa", "culpada",
        "dificuldade de amamentar", "dificuldade", "solidao", "sozinha",
    ]

    _LIMIAR_ENERGIA_BAIXA = 0.25
    _LIMIAR_TAXA_PAUSA = 0.45
    _LIMIAR_PAUSA_LONGA = 2.5
    _LIMIAR_FALA_LENTA = 1.5

    def calcular(self, features: FeaturesAcusticas, transcricao: str) -> Tuple[float, List[str]]:
        pontos = 0.0
        alertas: List[str] = []

        if features.energia_media < self._LIMIAR_ENERGIA_BAIXA:
            pontos += 0.30
            alertas.append("Energia vocal reduzida (hipofonia) - característica de depressão")

        if features.taxa_pausa > self._LIMIAR_TAXA_PAUSA:
            pontos += 0.20
            alertas.append(
                f"Alta proporção de silêncio ({features.taxa_pausa * 100:.0f}%) - "
                "sugestivo de retardo psicomotor"
            )

        if features.pausa_maxima_s > self._LIMIAR_PAUSA_LONGA:
            pontos += 0.15
            alertas.append(
                f"Pausa longa detectada ({features.pausa_maxima_s:.1f}s) - "
                "associada a estados dissociativos"
            )

        if 0 < features.taxa_fala_silabas_s < self._LIMIAR_FALA_LENTA:
            pontos += 0.15
            alertas.append("Fala lenta detectada - indicador de depressão")

        if features.pitch_desvio_hz < 15 and features.pitch_medio_hz > 0:
            pontos += 0.10
            alertas.append("Prosódia monotona - associada a afeto plano em depressão")

        texto = transcricao.lower()
        palavras_encontradas = [p for p in self._PALAVRAS_DEPRESSAO if p in texto]
        if palavras_encontradas:
            pontos += min(0.10 * len(palavras_encontradas), 0.30)
            alertas.append(
                f"Vocabulário associado a depressão detectado na transcrição: "
                f"{', '.join(palavras_encontradas[:3])}"
            )

        return min(pontos, 1.0), alertas


class ClassificadorRiscoAnsiedade:
    """
    Estima probabilidade de ansiedade gestacional.

    Indicadores:
    - Alta variação de pitch (voz instável)
    - Fala acelerada
    - Energia alta combinada com muitas pausas curtas (fala entrecortada)
    """

    _PALAVRAS_ANSIEDADE = [
        "preocupada", "preocupacao", "medo", "nervosa", "nervosismo",
        "nao consigo respirar", "coracao acelerado", "palpitacao",
        "nao durmo", "pesadelo", "estressada", "estresse",
        "e se", "e normal", "algo errado", "bebe", "beba", "nao sinto",
    ]

    _LIMIAR_PITCH_DESVIO = 40.0
    _LIMIAR_FALA_RAPIDA = 5.0

    def calcular(self, features: FeaturesAcusticas, transcricao: str) -> Tuple[float, List[str]]:
        pontos = 0.0
        alertas: List[str] = []

        if features.pitch_desvio_hz > self._LIMIAR_PITCH_DESVIO:
            pontos += 0.30
            alertas.append(
                f"Variação de pitch elevada ({features.pitch_desvio_hz:.0f} Hz std) - "
                "indicador de instabilidade emocional"
            )

        if features.taxa_fala_silabas_s > self._LIMIAR_FALA_RAPIDA:
            pontos += 0.25
            alertas.append("Fala acelerada - padrão associado a ansiedade")

        if features.energia_media > 0.7 and features.taxa_pausa > 0.2:
            pontos += 0.20
            alertas.append("Fala entrecortada com alta energia - indicador de agitação")

        texto = transcricao.lower()
        palavras_encontradas = [p for p in self._PALAVRAS_ANSIEDADE if p in texto]
        if palavras_encontradas:
            pontos += min(0.10 * len(palavras_encontradas), 0.30)
            alertas.append(
                f"Conteúdo verbal associado a ansiedade: "
                f"{', '.join(palavras_encontradas[:3])}"
            )

        return min(pontos, 1.0), alertas


class ClassificadorRiscoViolencia:
    """
    Detecta padrões vocais indicativos de trauma e violência doméstica.

    Indicadores (baseados em literatura de forensic phonetics):
    - Voz muito baixa (energia < limiar) com pausas longas (medo/coerção)
    - Tremor vocal (variação de pitch rápida em frequências baixas)
    - Hesitação excessiva antes de responder
    - Vocabulário evasivo ou de minimização
    """

    _PALAVRAS_VIOLENCIA = [
        "nao foi nada", "estou bem", "ele nao", "nao e assim",
        "cai", "bati", "machuquei", "nao lembro", "nao sei",
        "ele e bom", "foi sem querer", "minha culpa", "minha culpa",
        "exagerada", "exagerei", "ninguem precisa saber",
        "nao pode saber", "nao conte",
    ]

    _PALAVRAS_ALARME_DIRETO = [
        "me bate", "me bateu", "me agrediu", "me xinga", "me ameaca",
        "tenho medo dele", "medo do meu marido", "medo do meu parceiro",
        "me machuca", "me machucar", "nao me deixa sair",
    ]

    _LIMIAR_ENERGIA_MUITO_BAIXA = 0.15

    def calcular(self, features: FeaturesAcusticas, transcricao: str) -> Tuple[float, List[str]]:
        pontos = 0.0
        alertas: List[str] = []
        texto = transcricao.lower()

        # Relato direto: maior peso
        palavras_alarme = [p for p in self._PALAVRAS_ALARME_DIRETO if p in texto]
        if palavras_alarme:
            pontos += 0.70
            alertas.append(
                "ALERTA CRÍTICO: linguagem indicativa de violência identificada na fala. "
                "Acionar protocolo de acolhimento imediato."
            )

        if features.energia_media < self._LIMIAR_ENERGIA_MUITO_BAIXA:
            pontos += 0.20
            alertas.append(
                "Voz extremamente baixa durante o relato - possivelmente associada a medo ou coerção"
            )

        if features.pausa_maxima_s > 3.5:
            pontos += 0.15
            alertas.append(
                f"Pausa prolongada antes de responder ({features.pausa_maxima_s:.1f}s) - "
                "possível hesitação por medo de represália"
            )

        palavras_evasivas = [p for p in self._PALAVRAS_VIOLENCIA if p in texto]
        if palavras_evasivas:
            pontos += min(0.08 * len(palavras_evasivas), 0.25)
            alertas.append(
                f"Linguagem evasiva ou de minimização detectada: "
                f"{', '.join(palavras_evasivas[:3])}"
            )

        return min(pontos, 1.0), alertas


class ClassificadorFadiga:
    """Detecta fadiga vocal associada a fadiga hormonal pós-parto."""

    def calcular(self, features: FeaturesAcusticas, transcricao: str) -> Tuple[float, List[str]]:
        pontos = 0.0
        alertas: List[str] = []

        if features.energia_media < 0.20:
            pontos += 0.40
            alertas.append("Energia vocal muito reduzida - indicador de fadiga")

        if features.taxa_fala_silabas_s < 1.0 and features.taxa_fala_silabas_s > 0:
            pontos += 0.30
            alertas.append("Fala muito lenta - associada a fadiga física ou hormonal")

        if "cansada" in transcricao.lower() or "exausta" in transcricao.lower():
            pontos += 0.30
            alertas.append("Relato verbal de cansaço ou exaustão")

        return min(pontos, 1.0), alertas


# ---------------------------------------------------------------------------
# Pipeline principal de analise de audio
# ---------------------------------------------------------------------------

class AnalisadorAudio:
    """
    Pipeline principal de análise de áudio para saúde da mulher.

    Fluxo:
    1. Extrair features acústicas (librosa ou numpy)
    2. Transcrever áudio (Whisper ou vazio)
    3. Calcular pontuações de risco (depressão, ansiedade, violência, fadiga)
    4. Consolidar resultado e gerar resumo clínico
    """

    def __init__(self):
        self._extrator = ExtratorFeatures()
        self._transcritor = TranscritorAudio()
        self._clf_depressao = ClassificadorRiscoDepressao()
        self._clf_ansiedade = ClassificadorRiscoAnsiedade()
        self._clf_violencia = ClassificadorRiscoViolencia()
        self._clf_fadiga = ClassificadorFadiga()

    def processar_arquivo(
        self, caminho: str, tipo: TipoConsulta
    ) -> ResultadoAudio:
        features, sinal, sr = self._extrator.extrair_de_arquivo(caminho)
        transcricao = self._transcritor.transcrever(sinal, sr)
        return self._calcular_resultado(features, transcricao, tipo)

    def processar_sintetico(self, tipo: TipoConsulta) -> ResultadoAudio:
        """
        Processa áudio sintético para demonstração.

        Gera um sinal com características controladas de acordo com o tipo de consulta,
        permitindo validar o pipeline sem necessidade de gravações reais de pacientes.
        """
        sinal, sr = self._gerar_audio_sintetico(tipo)
        features = self._extrator.extrair_de_array(sinal, sr)
        transcricao = self._gerar_transcricao_sintetica(tipo)
        return self._calcular_resultado(features, transcricao, tipo)

    def _gerar_audio_sintetico(
        self, tipo: TipoConsulta
    ) -> Tuple[np.ndarray, int]:
        sr = 16000
        duracao_s = 15
        t = np.linspace(0, duracao_s, sr * duracao_s)
        rng = np.random.default_rng(0)

        if tipo == TipoConsulta.POS_PARTO:
            # Áudio de baixa energia, lento, monotono
            freq_base = 150.0
            sinal = 0.1 * np.sin(2 * np.pi * freq_base * t)
            silencio = np.zeros(sr * 3)
            sinal = np.concatenate([sinal[:sr * 5], silencio, sinal[sr * 5:sr * 10]])

        elif tipo == TipoConsulta.PRE_NATAL:
            # Áudio com pitch instável (ansiedade)
            freq_mod = 200.0 + 50.0 * np.sin(2 * np.pi * 0.5 * t)
            sinal = 0.6 * np.sin(2 * np.pi * freq_mod * t)
            sinal += 0.05 * rng.standard_normal(len(sinal))

        elif tipo == TipoConsulta.TRIAGEM_VIOLENCIA:
            # Áudio muito baixo com pausa longa
            sinal = 0.05 * np.sin(2 * np.pi * 180.0 * t)
            pausa = np.zeros(sr * 4)
            sinal = np.concatenate([sinal[:sr * 3], pausa, sinal[sr * 3:]])

        else:
            sinal = 0.4 * np.sin(2 * np.pi * 220.0 * t)
            sinal += 0.02 * rng.standard_normal(len(sinal))

        return sinal.astype(np.float32), sr

    def _gerar_transcricao_sintetica(self, tipo: TipoConsulta) -> str:
        transcricoes = {
            TipoConsulta.GINECOLOGICA: (
                "Doutora, estou sentindo uma dor na região pélvica há dois dias. "
                "Não sei se é normal, fico preocupada."
            ),
            TipoConsulta.PRE_NATAL: (
                "Fico muito preocupada com o bebê. E se algo estiver errado? "
                "Não consigo dormir direito. Sinto palpitão às vezes."
            ),
            TipoConsulta.POS_PARTO: (
                "Estou muito cansada. Fico chorando sem motivo. "
                "Não tenho vontade de sair. Me sinto sozinha mesmo com o bebê."
            ),
            TipoConsulta.TRIAGEM_VIOLENCIA: (
                "Não foi nada. Caí da escada. Estou bem. Ele é bom, foi sem querer. "
                "Por favor não conte para ninguém."
            ),
        }
        return transcricoes.get(tipo, "")

    def _calcular_resultado(
        self,
        features: FeaturesAcusticas,
        transcricao: str,
        tipo: TipoConsulta,
    ) -> ResultadoAudio:
        p_dep, a_dep = self._clf_depressao.calcular(features, transcricao)
        p_ans, a_ans = self._clf_ansiedade.calcular(features, transcricao)
        p_vio, a_vio = self._clf_violencia.calcular(features, transcricao)
        p_fad, a_fad = self._clf_fadiga.calcular(features, transcricao)

        # Pesos por tipo de consulta
        pesos = {
            TipoConsulta.GINECOLOGICA: (0.2, 0.3, 0.2, 0.3),
            TipoConsulta.PRE_NATAL: (0.3, 0.5, 0.1, 0.1),
            TipoConsulta.POS_PARTO: (0.5, 0.2, 0.1, 0.2),
            TipoConsulta.TRIAGEM_VIOLENCIA: (0.1, 0.1, 0.7, 0.1),
        }
        w_dep, w_ans, w_vio, w_fad = pesos.get(tipo, (0.25, 0.25, 0.25, 0.25))
        risco_geral = w_dep * p_dep + w_ans * p_ans + w_vio * p_vio + w_fad * p_fad

        todos_alertas = a_dep + a_ans + a_vio + a_fad
        resumo = self._gerar_resumo_clinico(tipo, p_dep, p_ans, p_vio, p_fad, risco_geral)

        return ResultadoAudio(
            tipo_consulta=tipo,
            duracao_s=features.duracao_s,
            features=features,
            transcricao=transcricao,
            pontuacao_depressao=round(p_dep, 4),
            pontuacao_ansiedade=round(p_ans, 4),
            pontuacao_violencia=round(p_vio, 4),
            pontuacao_fadiga=round(p_fad, 4),
            pontuacao_risco_geral=round(risco_geral, 4),
            alertas=todos_alertas,
            resumo_clinico=resumo,
        )

    def _gerar_resumo_clinico(
        self,
        tipo: TipoConsulta,
        p_dep: float,
        p_ans: float,
        p_vio: float,
        p_fad: float,
        risco_geral: float,
    ) -> str:
        nivel = "BAIXO" if risco_geral < 0.25 else ("MODERADO" if risco_geral < 0.55 else "ALTO")

        linhas = [
            f"Análise de áudio - {tipo.value.replace('_', ' ').title()}.",
            f"Risco geral: {nivel} ({risco_geral:.2f}).",
        ]

        if p_dep > 0.3:
            linhas.append(f"  - Indicadores de depressão: {p_dep:.2f} - recomenda-se triagem formal (EPDS).")
        if p_ans > 0.3:
            linhas.append(f"  - Indicadores de ansiedade: {p_ans:.2f} - considerar encaminhamento psicológico.")
        if p_vio > 0.3:
            linhas.append(
                f"  - Indicadores de violência: {p_vio:.2f} - ACIONAR PROTOCOLO DE ACOLHIMENTO."
            )
        if p_fad > 0.3:
            linhas.append(f"  - Indicadores de fadiga: {p_fad:.2f} - avaliar quadro hormonal e sono.")

        return " ".join(linhas)
