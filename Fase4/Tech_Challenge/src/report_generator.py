"""
Módulo de geração de relatórios clínicos automatizados.

Gera relatórios em formato texto estruturado e JSON, contendo:
- Resumo executivo da análise multimodal
- Detalhamento dos resultados por modalidade (vídeo, áudio)
- Alertas classificados por nível de urgência
- Recomendações de conduta clínica
- Metadados de rastreabilidade (timestamp, hash de auditoria)

Todos os relatórios são anonimizados antes da persistência.
Nenhum dado identificável de paciente é gravado nos arquivos de saída.

Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from audio_analysis import ResultadoAudio
from multimodal_fusion import RiscoMultimodal
from video_analysis import ResultadoVideo

logger = logging.getLogger(__name__)


class GeradorRelatorio:
    """
    Gera relatórios clínicos a partir dos resultados de cada modalidade
    e do resultado de fusão multimodal.

    Os relatórios gerados não contêm dados que permitam identificar
    individualmente a paciente. O ID da paciente é substituído por um
    hash truncado de 12 caracteres antes de qualquer escrita em disco.
    """

    def __init__(self, diretorio_saida: str = "results/relatorios"):
        self.diretorio = Path(diretorio_saida)
        self.diretorio.mkdir(parents=True, exist_ok=True)

    def _anonimizar_id(self, paciente_id: str) -> str:
        return hashlib.sha256(paciente_id.encode()).hexdigest()[:12]

    def _timestamp_iso(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def gerar(
        self,
        paciente_id: str,
        tipo_atendimento: str,
        resultado_video: Optional[ResultadoVideo],
        resultado_audio: Optional[ResultadoAudio],
        risco_multimodal: RiscoMultimodal,
        salvar: bool = True,
    ) -> dict:
        """
        Gera o relatório completo e opcionalmente salva em disco.

        Retorna o dicionário do relatório (independente de salvar).
        """
        id_anonimo = self._anonimizar_id(paciente_id)
        timestamp = self._timestamp_iso()

        relatorio = {
            "paciente_id_anonimo": id_anonimo,
            "timestamp_utc": timestamp,
            "tipo_atendimento": tipo_atendimento,
            "nivel_prioridade": risco_multimodal.nivel_prioridade.value,
            "pontuacao_risco_fusao": risco_multimodal.pontuacao_fusao,
            "resumo_executivo": risco_multimodal.resumo_executivo,
            "alertas_criticos": risco_multimodal.alertas_criticos,
            "alertas_moderados": risco_multimodal.alertas_moderados,
            "recomendacoes": risco_multimodal.recomendacoes,
            "detalhe_video": self._serializar_video(resultado_video),
            "detalhe_audio": self._serializar_audio(resultado_audio),
        }

        if salvar:
            nome_arquivo = f"relatorio_{id_anonimo}_{timestamp.replace(':', '').replace('-', '')}.json"
            caminho = self.diretorio / nome_arquivo
            with open(caminho, "w", encoding="utf-8") as f:
                json.dump(relatorio, f, ensure_ascii=False, indent=2)
            logger.info("Relatório salvo em: %s", caminho)

        return relatorio

    def _serializar_video(self, resultado: Optional[ResultadoVideo]) -> Optional[dict]:
        if resultado is None:
            return None
        return {
            "tipo": resultado.tipo.value,
            "total_frames": resultado.total_frames,
            "frames_processados": resultado.frames_processados,
            "duracao_s": resultado.duracao_s,
            "pontuacao_risco_media": resultado.pontuacao_risco_media,
            "pontuacao_risco_maxima": resultado.pontuacao_risco_maxima,
            "alertas_criticos": resultado.alertas_criticos,
            "resumo_clinico": resultado.resumo_clinico,
            "n_frames_com_alerta": len(resultado.frames_com_alerta),
        }

    def _serializar_audio(self, resultado: Optional[ResultadoAudio]) -> Optional[dict]:
        if resultado is None:
            return None
        return {
            "tipo_consulta": resultado.tipo_consulta.value,
            "duracao_s": resultado.duracao_s,
            "transcricao_disponivel": bool(resultado.transcricao),
            "pontuacao_depressao": resultado.pontuacao_depressao,
            "pontuacao_ansiedade": resultado.pontuacao_ansiedade,
            "pontuacao_violencia": resultado.pontuacao_violencia,
            "pontuacao_fadiga": resultado.pontuacao_fadiga,
            "pontuacao_risco_geral": resultado.pontuacao_risco_geral,
            "alertas": resultado.alertas,
            "resumo_clinico": resultado.resumo_clinico,
            "features_acusticas": {
                "energia_media": resultado.features.energia_media,
                "pitch_medio_hz": resultado.features.pitch_medio_hz,
                "pitch_desvio_hz": resultado.features.pitch_desvio_hz,
                "taxa_pausa": resultado.features.taxa_pausa,
                "pausa_maxima_s": resultado.features.pausa_maxima_s,
                "taxa_fala_silabas_s": resultado.features.taxa_fala_silabas_s,
            },
        }

    def imprimir_relatorio(self, relatorio: dict) -> None:
        """Imprime o relatório de forma legível no terminal."""
        nivel = relatorio.get("nivel_prioridade", "N/A")
        separador = "=" * 60

        print(separador)
        print(f"RELATÓRIO CLÍNICO MULTIMODAL")
        print(f"Paciente (anônimo): {relatorio.get('paciente_id_anonimo')}")
        print(f"Timestamp: {relatorio.get('timestamp_utc')}")
        print(f"Tipo de atendimento: {relatorio.get('tipo_atendimento')}")
        print(separador)
        print(f"NÍVEL DE PRIORIDADE: {nivel}")
        print(f"Pontuação de risco fusionada: {relatorio.get('pontuacao_risco_fusao', 0):.2f}")
        print()
        print("RESUMO EXECUTIVO:")
        print(f"  {relatorio.get('resumo_executivo', '')}")
        print()

        alertas_criticos = relatorio.get("alertas_criticos", [])
        if alertas_criticos:
            print("ALERTAS CRÍTICOS:")
            for alerta in alertas_criticos:
                print(f"  [!] {alerta}")
            print()

        alertas_moderados = relatorio.get("alertas_moderados", [])
        if alertas_moderados:
            print("ALERTAS MODERADOS:")
            for alerta in alertas_moderados[:5]:
                print(f"  [-] {alerta}")
            if len(alertas_moderados) > 5:
                print(f"  ... e mais {len(alertas_moderados) - 5} alertas.")
            print()

        recomendacoes = relatorio.get("recomendacoes", [])
        if recomendacoes:
            print("RECOMENDAÇÕES DE CONDUTA:")
            for rec in recomendacoes:
                print(f"  > {rec}")
            print()

        detalhe_video = relatorio.get("detalhe_video")
        if detalhe_video:
            print("ANÁLISE DE VÍDEO:")
            print(f"  Tipo: {detalhe_video.get('tipo')}")
            print(f"  Risco médio: {detalhe_video.get('pontuacao_risco_media', 0):.2f}")
            print(f"  Risco máximo: {detalhe_video.get('pontuacao_risco_maxima', 0):.2f}")
            print(f"  Frames com alerta: {detalhe_video.get('n_frames_com_alerta', 0)}")
            print()

        detalhe_audio = relatorio.get("detalhe_audio")
        if detalhe_audio:
            print("ANÁLISE DE ÁUDIO:")
            print(f"  Tipo de consulta: {detalhe_audio.get('tipo_consulta')}")
            print(f"  Depressão: {detalhe_audio.get('pontuacao_depressao', 0):.2f}")
            print(f"  Ansiedade: {detalhe_audio.get('pontuacao_ansiedade', 0):.2f}")
            print(f"  Violência: {detalhe_audio.get('pontuacao_violencia', 0):.2f}")
            print(f"  Fadiga: {detalhe_audio.get('pontuacao_fadiga', 0):.2f}")
            print()

        print(separador)
        print("Aviso: Este relatório é uma ferramenta de apoio clínico.")
        print("A decisão final é sempre responsabilidade do profissional de saúde.")
        print(separador)
