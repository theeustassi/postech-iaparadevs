"""
Assistente Medico Virtual com LangChain (RAG + LLM).

Arquitetura:
    Pergunta do usuario
        -> Retrieval (FAISS + sentence-transformers)  <- Busca contexto relevante
        -> Prompt montado com contexto + pergunta
        -> LLM (Google Gemini ou modelo local HuggingFace)
        -> Resposta com fonte indicada

O LLM padrao e o Google Gemini (gratuito via API), com fallback automatico
para o modelo fine-tunado local se a API nao estiver disponivel.

Seguranca:
    - Prompt de sistema com restricoes explicitas
    - Nunca prescreve medicamentos diretamente
    - Sempre indica a fonte da resposta
    - Respostas marcadas com aviso de responsabilidade clinica
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Voce e um assistente medico virtual de apoio do HospitalIQ.

Voce auxilia medicos e profissionais de saude com base nos protocolos internos do hospital.

REGRAS OBRIGATORIAS (nao podem ser ignoradas):
1. NUNCA prescrevera medicamentos diretamente. Use sempre: "sugere-se avaliar com o medico responsavel".
2. NUNCA substitua a avaliacao clinica presencial.
3. Indique SEMPRE a fonte do protocolo utilizado na sua resposta.
4. Em situacoes de risco de vida, oriente IMEDIATAMENTE acionar a equipe de emergencia.
5. Responda em portugues brasileiro de forma clara e objetiva.

Voce pode:
- Explicar protocolos e condutas clinicas do hospital
- Auxiliar na interpretacao de criterios diagnosticos
- Sugerir exames com base na queixa apresentada
- Informar sobre escores clinicos (CURB-65, SOFA, Wells, etc.)
"""

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]


class MedicalAssistant:
    """
    Assistente medico baseado em RAG (Retrieval-Augmented Generation).

    Construcao da base de conhecimento:
        assistant = MedicalAssistant(api_key="...")
        assistant.build_knowledge_base(["data/medical_protocols.txt", "data/pubmedqa_train.jsonl"])
        assistant.save_knowledge_base()

    Uso em producao (com base ja salva):
        assistant = MedicalAssistant(api_key="...")
        assistant.load_knowledge_base()
        assistant.build_rag_chain()
        resultado = assistant.ask("Qual o tratamento de sepse?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.preferred_model = preferred_model
        self.vectorstore = None
        self.rag_chain = None
        self.llm = None
        self.llm_backend: str = "nao_inicializado"

    # ------------------------------------------------------------------
    # Inicializacao do LLM
    # ------------------------------------------------------------------

    def _init_gemini(self):
        """Tenta inicializar o Google Gemini com fallback entre modelos."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        models_to_try = (
            [self.preferred_model] if self.preferred_model else GEMINI_MODELS
        )

        for model in models_to_try:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=self.api_key,
                    temperature=0.3,
                    max_output_tokens=2048,
                )
                # Verifica se o modelo responde
                llm.invoke("ok")
                logger.info(f"LLM inicializado com Google Gemini: {model}")
                self.llm_backend = f"Google Gemini ({model})"
                return llm
            except Exception as e:
                logger.warning(f"Modelo Gemini '{model}' indisponivel: {e}")

        return None

    def _init_local_llm(self):
        """Carrega o modelo local (fine-tunado ou flan-t5-base como fallback)."""
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline

            finetuned_path = "results/modelos/finetuned_model"
            model_id = (
                finetuned_path if Path(finetuned_path).exists() else "google/flan-t5-base"
            )
            logger.info(f"Carregando modelo local: {model_id}")

            import torch
            use_gpu = torch.cuda.is_available()
            pipe = pipeline(
                "text2text-generation",
                model=model_id,
                max_new_tokens=512,
                temperature=0.3,
                device=0 if use_gpu else -1,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
            )
            self.llm_backend = f"Modelo local ({model_id.split('/')[-1]})"
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            raise RuntimeError(
                f"Nao foi possivel inicializar nenhum LLM. Verifique a API key ou instale "
                f"as dependencias locais (transformers, torch). Erro: {e}"
            )

    def _load_llm(self):
        """Inicializa o LLM: Gemini primeiro, fallback para modelo local."""
        if self.api_key:
            llm = self._init_gemini()
            if llm is not None:
                return llm

        logger.warning("Gemini nao disponivel. Usando modelo local como fallback.")
        return self._init_local_llm()

    # ------------------------------------------------------------------
    # Construcao da base de conhecimento vetorial
    # ------------------------------------------------------------------

    def _generate_protocols_doc(self, fallback_path: str) -> list:
        """
        Gera protocolos clinicos atualizados via Gemini e retorna como Documents
        para indexacao na base vetorial FAISS.

        Se o Gemini nao estiver disponivel ou falhar, carrega o arquivo local
        como fallback automatico.

        Args:
            fallback_path: caminho para o arquivo .txt usado como fallback

        Returns:
            lista de Documents prontos para indexacao no FAISS
        """
        from langchain_core.documents import Document

        if self.api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm_gen = None
                for model in GEMINI_MODELS:
                    try:
                        llm_gen = ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=self.api_key,
                            temperature=0.2,
                            max_output_tokens=8192,
                        )
                        break
                    except Exception:
                        llm_gen = None

                if llm_gen is None:
                    raise RuntimeError(
                        "Nenhum modelo Gemini disponivel para geracao de protocolos"
                    )

                PROTOCOL_PROMPT = (
                    "Voce e um medico especialista e gerente de qualidade hospitalar. "
                    "Elabore um documento de protocolos clinicos hospitalares abrangente, "
                    "em portugues, cobrindo:\n"
                    "1. Sepse e choque septico (bundles de 1h/3h/6h, criterios Sepsis-3, SOFA)\n"
                    "2. Protocolo de ressuscitacao cardiopulmonar (ACLS, BLS, desfibrilador)\n"
                    "3. AVC isquemico: janela terapeutica, alteplase, trombectomia mecanica\n"
                    "4. Infarto agudo do miocardio STEMI: reperfusao, antiagregacao, IECA\n"
                    "5. Pneumonia adquirida na comunidade: CURB-65, antibioticos, criterios de internacao\n"
                    "6. Triagem hospitalar: Protocolo de Manchester, 5 niveis de prioridade\n"
                    "7. Cetoacidose diabetica: criterios, hidratacao, infusao de insulina\n"
                    "8. Fibrilacao atrial: CHA2DS2-VASc, cardioversao, anticoagulacao\n"
                    "Para cada protocolo inclua: definicao/criterios diagnosticos, manejo "
                    "terapeutico passo a passo, indicacoes de exames, alertas de seguranca "
                    "e principais referencias. Linguagem tecnica para profissionais de saude."
                )

                content = llm_gen.invoke(PROTOCOL_PROMPT).content
                logger.info("Protocolos clinicos gerados via Gemini para base de conhecimento.")
                return [
                    Document(
                        page_content=content,
                        metadata={
                            "source": "Protocolos Clinicos (gerado por Gemini)",
                            "generated": "true",
                        },
                    )
                ]

            except Exception as e:
                logger.warning(
                    f"Nao foi possivel gerar protocolos via LLM ({e}). "
                    f"Usando arquivo local como fallback: {fallback_path}"
                )

        # Fallback: arquivo local
        try:
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(fallback_path, encoding="utf-8")
            docs = loader.load()
            logger.info(f"Protocolos carregados do arquivo local: {fallback_path}")
            return docs
        except Exception as e:
            logger.error(
                f"Erro ao carregar arquivo de fallback '{fallback_path}': {e}"
            )
            return []

    def build_knowledge_base(self, documents_paths: List[str]) -> None:
        """
        Cria a base vetorial (FAISS) a partir dos documentos fornecidos.

        Os embeddings sao gerados com o modelo sentence-transformers/all-MiniLM-L6-v2,
        que roda localmente em CPU sem necessidade de API key.

        Args:
            documents_paths: lista de caminhos para arquivos .txt ou .jsonl
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        all_docs = []

        for path in documents_paths:
            try:
                if path.endswith(".txt"):
                    all_docs.extend(self._generate_protocols_doc(path))

                elif path.endswith(".jsonl"):
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            content = (
                                f"Pergunta: {rec.get('question', '')}\n"
                                f"Contexto: {rec.get('context', '')}\n"
                                f"Resposta: {rec.get('answer', '')}"
                            )
                            all_docs.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": rec.get("source", path),
                                        "id": str(rec.get("id", "")),
                                    },
                                )
                            )
            except Exception as e:
                logger.warning(f"Erro ao carregar '{path}': {e}")

        if not all_docs:
            raise ValueError(
                "Nenhum documento carregado. Verifique os caminhos fornecidos."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
        )
        chunks = splitter.split_documents(all_docs)
        logger.info(
            f"Base de conhecimento: {len(all_docs)} documentos -> {len(chunks)} chunks"
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if __import__('torch').cuda.is_available() else "cpu"},
        )
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("Base vetorial (FAISS) criada com sucesso.")

    def save_knowledge_base(
        self, path: str = "results/modelos/vectorstore"
    ) -> None:
        """Salva a base vetorial em disco para reuso."""
        if self.vectorstore is None:
            raise RuntimeError("Base vetorial nao inicializada.")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(path)
        logger.info(f"Base vetorial salva em: {path}")

    def load_knowledge_base(
        self, path: str = "results/modelos/vectorstore"
    ) -> None:
        """Carrega uma base vetorial previamente salva."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if __import__('torch').cuda.is_available() else "cpu"},
        )
        self.vectorstore = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Base vetorial carregada de: {path}")

    # ------------------------------------------------------------------
    # Pipeline RAG
    # ------------------------------------------------------------------

    def build_rag_chain(self) -> None:
        """
        Monta o pipeline RAG usando LCEL (LangChain Expression Language).

        Fluxo:
            pergunta -> retriever (busca k=3 documentos mais relevantes)
                     -> formata contexto com fontes
                     -> monta prompt (system + contexto + pergunta)
                     -> LLM gera resposta
                     -> StrOutputParser extrai o texto final
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Base vetorial nao inicializada. "
                "Execute build_knowledge_base() ou load_knowledge_base() primeiro."
            )

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough

        self.llm = self._load_llm()

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3},
        )

        def format_context_with_sources(docs) -> str:
            """Formata os documentos recuperados incluindo a fonte de cada trecho."""
            parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "protocolo interno")
                parts.append(f"[Fonte {i} - {source}]\n{doc.page_content}")
            return "\n\n---\n\n".join(parts)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "Contexto dos protocolos internos:\n{context}\n\n"
                        "Pergunta:\n{question}"
                    ),
                ),
            ]
        )

        self.rag_chain = (
            {
                "context": retriever | format_context_with_sources,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("Pipeline RAG montado e pronto para uso.")

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        patient_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Consulta o assistente medico.

        Args:
            question: pergunta do profissional de saude
            patient_context: contexto adicional do paciente (opcional)

        Returns:
            dict com:
                - response: resposta gerada pelo LLM
                - sources: lista de fontes consultadas
                - safety_disclaimer: aviso obrigatorio de responsabilidade
        """
        if self.rag_chain is None:
            raise RuntimeError(
                "Pipeline nao inicializado. Execute build_rag_chain() primeiro."
            )

        full_question = question
        if patient_context:
            full_question = f"Contexto do paciente: {patient_context}\n\nPergunta: {question}"

        # Recupera os documentos para informar as fontes
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        source_docs = retriever.invoke(question)
        sources = list(
            {doc.metadata.get("source", "protocolo interno") for doc in source_docs}
        )

        response = self.rag_chain.invoke(full_question)

        return {
            "response": response,
            "sources": sources,
            "llm_backend": self.llm_backend,
            "safety_disclaimer": (
                "AVISO: Esta resposta e de carater informativo e de apoio ao profissional de saude. "
                "A decisao clinica final e de responsabilidade exclusiva do medico assistente."
            ),
        }
