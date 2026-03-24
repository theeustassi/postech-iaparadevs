"""
Pipeline de fine-tuning de LLM com LoRA/PEFT para dados medicos.

Modelos suportados:
  - google/flan-t5-base          (250M, seq2seq, CPU/GPU, demo rapido)
  - microsoft/Phi-3-mini-4k-instruct  (3.8B, causal, fp16 em 16GB VRAM) [RECOMENDADO]
  - BioMistral/BioMistral-7B     (7B, causal, medico especializado, use_4bit=True)
  - meta-llama/Llama-3.2-3B-Instruct  (3B, causal, bom custo-beneficio)

Tecnica: Parameter-Efficient Fine-Tuning (PEFT) com LoRA ou QLoRA (4-bit)
  - Congela os pesos originais do modelo
  - Treina apenas matrizes de baixo rank adicionadas as camadas de atencao
  - Resultado: apenas ~0.1-1% dos parametros sao treinaveis
  - QLoRA (use_4bit=True): quantiza o modelo base em 4-bit antes do LoRA,
    reduzindo VRAM necessaria para modelos 7B+ de ~14GB para ~4GB

Dependencias necessarias:
  pip install transformers peft trl accelerate datasets torch
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """
    Configuracoes do pipeline de fine-tuning.

    Para hardware limitado (CPU): reduza max_steps e batch_size.
    Para GPU: aumente num_train_epochs e desabilite max_steps.
    """

    # Modelo base do HuggingFace Hub
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"

    # Diretorio para salvar o modelo fine-tunado
    output_dir: str = "results/modelos/finetuned_model"

    # Controle de treinamento
    max_steps: int = -1           # -1 = usa num_train_epochs (recomendado com GPU)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4    # Phi-3-mini (3.8B): 4 e seguro em 16GB
    gradient_accumulation_steps: int = 4   # Batch efetivo = 4 * 4 = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.05

    # Configuracoes LoRA
    lora_r: int = 16              # Rank das matrizes de baixo rank
    lora_alpha: int = 32          # Escala de aprendizado LoRA
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # None = detecta automaticamente

    # QLoRA: quantizacao 4-bit (recomendado para modelos 7B+)
    # Exige: pip install bitsandbytes
    use_4bit: bool = False


class MedicalLLMFineTuner:
    """
    Pipeline completo de fine-tuning de LLM para dados medicos.

    Utiliza PEFT (LoRA) + TRL (SFTTrainer) para um treinamento eficiente
    mesmo com recursos computacionais limitados.

    Exemplo de uso:
        config = FineTuningConfig(model_name="google/flan-t5-base", max_steps=50)
        tuner = MedicalLLMFineTuner(config)
        tuner.load_model()
        tuner.train(train_records, val_records)
    """

    def __init__(self, config: Optional[FineTuningConfig] = None):
        self.config = config or FineTuningConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _check_dependencies(self) -> bool:
        """Verifica se as dependencias de treinamento estao instaladas."""
        try:
            import transformers  # noqa: F401
            import peft          # noqa: F401
            import trl           # noqa: F401
            import torch         # noqa: F401
            return True
        except ImportError as e:
            logger.error(f"Dependencia nao encontrada: {e}")
            logger.error("Execute: pip install transformers peft trl torch accelerate")
            return False

    def _is_seq2seq_model(self) -> bool:
        """Verifica se o modelo e do tipo encoder-decoder (seq2seq)."""
        seq2seq_keywords = ["t5", "bart", "pegasus", "mbart", "mt5"]
        return any(k in self.config.model_name.lower() for k in seq2seq_keywords)

    def load_model(self) -> None:
        """
        Carrega o modelo base e aplica a configuracao LoRA.

        O modelo original e congelado; apenas os adaptadores LoRA sao treinaveis.
        """
        if not self._check_dependencies():
            raise ImportError("Instale as dependencias necessarias antes de continuar.")

        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        use_gpu = torch.cuda.is_available()
        device_info = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        dtype = torch.float16 if use_gpu else torch.float32
        logger.info(f"Dispositivo detectado: {device_info}")
        logger.info(f"Carregando modelo base: {self.config.model_name} (dtype={dtype})")

        # QLoRA: configura quantizacao 4-bit antes de carregar o modelo
        bnb_config = None
        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("QLoRA ativo: modelo sera carregado em 4-bit (NF4 + double quant)")
            except ImportError:
                logger.warning(
                    "bitsandbytes nao instalado. use_4bit ignorado. "
                    "Execute: pip install bitsandbytes"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Carrega o modelo conforme o tipo (seq2seq ou causal)
        if self._is_seq2seq_model():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                device_map="auto" if use_gpu else None,
            )
            task_type = TaskType.SEQ_2_SEQ_LM
            target_modules = self.config.lora_target_modules or ["q", "v"]
        else:
            load_kwargs = dict(
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if use_gpu else None,
            )
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
                # Com QLoRA, device_map="auto" e obrigatorio
                load_kwargs["device_map"] = "auto"
                load_kwargs.pop("torch_dtype", None)

            # Phi-3 compatibility: versoes do modeling_phi3.py cacheado no HuggingFace
            # aceitam apenas rope_scaling=None ou rope_scaling["type"]=="longrope".
            # - Configs recentes usam "rope_type" em vez de "type" -> KeyError
            # - Quando rope_type=="default" (RoPE padrao) o modelo nao precisa de
            #   escalonamento; remover rope_scaling evita ValueError na inicializacao.
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(
                    self.config.model_name, trust_remote_code=True
                )
                rs = getattr(model_cfg, "rope_scaling", None)
                if isinstance(rs, dict):
                    rope_type = rs.get("rope_type") or rs.get("type", "")
                    if rope_type in ("default", ""):
                        # RoPE padrao: remove o dict para evitar ValueError no cacheado
                        model_cfg.rope_scaling = None
                        logger.info("rope_scaling removido (tipo 'default') para compatibilidade")
                    elif "type" not in rs:
                        rs["type"] = rope_type
                        logger.info(f"rope_scaling['type'] = '{rope_type}' injetado para compatibilidade")
                load_kwargs["config"] = model_cfg
            except Exception as _cfg_exc:
                logger.debug(f"Pre-load de config ignorado: {_cfg_exc}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )
            task_type = TaskType.CAUSAL_LM
            if self.config.lora_target_modules:
                target_modules = self.config.lora_target_modules
            else:
                # Detecta os nomes reais das projecoes de atencao no modelo carregado.
                # Phi-3 usa qkv_proj (Q+K+V combinados); a maioria dos outros usa q_proj/v_proj.
                all_names = {name.split(".")[-1] for name, _ in self.model.named_modules()}
                if "qkv_proj" in all_names:
                    target_modules = ["qkv_proj", "o_proj"]
                elif "q_proj" in all_names:
                    target_modules = ["q_proj", "v_proj"]
                else:
                    target_modules = "all-linear"
                logger.info(f"target_modules detectados automaticamente: {target_modules}")

            # QLoRA exige prepare_model_for_kbit_training
            if bnb_config is not None:
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)

        # Configura e aplica o LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=task_type,
        )
        self.model = get_peft_model(self.model, lora_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Parametros treinaveis (LoRA): {trainable:,} de {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def prepare_dataset(self, formatted_records: List[Dict[str, str]]):
        """Converte a lista de dicionarios em um HuggingFace Dataset."""
        from datasets import Dataset

        texts = [r["text"] for r in formatted_records]
        return Dataset.from_dict({"text": texts})

    def train(
        self,
        train_records: List[Dict[str, str]],
        val_records: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Executa o fine-tuning usando SFTTrainer da biblioteca TRL.

        Args:
            train_records: lista de exemplos formatados para treino
            val_records: lista de exemplos formatados para validacao (opcional)

        Returns:
            objeto TrainOutput com metricas de treinamento
        """
        if self.model is None:
            self.load_model()

        from trl import SFTConfig, SFTTrainer

        train_dataset = self.prepare_dataset(train_records)
        eval_dataset = self.prepare_dataset(val_records) if val_records else None

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        import torch as _torch
        use_gpu = _torch.cuda.is_available()

        # TRL 0.29+: SFTConfig unifica TrainingArguments + parametros do SFTTrainer
        # max_length substitui max_seq_length; dataset_text_field permanece igual
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            report_to="none",
            fp16=use_gpu and not self.config.use_4bit,
            bf16=False,
            gradient_checkpointing=use_gpu,
            dataloader_num_workers=4 if use_gpu else 0,
            optim="paged_adamw_8bit" if self.config.use_4bit else (
                "adamw_torch_fused" if use_gpu else "adamw_torch"
            ),
            dataset_text_field="text",
            max_length=self.config.max_seq_length,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Iniciando fine-tuning com LoRA...")
        train_result = self.trainer.train()
        logger.info(
            f"Fine-tuning concluido. Loss final: {train_result.training_loss:.4f}"
        )

        self.save_model()
        return train_result

    def save_model(self) -> None:
        """Salva o modelo fine-tunado e o tokenizer."""
        if self.trainer is None:
            raise RuntimeError("Treine o modelo antes de salvar.")
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Modelo salvo em: {self.config.output_dir}")

    def load_finetuned(self, model_path: str) -> None:
        """
        Carrega um modelo previamente fine-tunado a partir de um diretorio local.

        Args:
            model_path: caminho para o diretorio com o modelo salvo
        """
        from transformers import AutoTokenizer

        try:
            from peft import AutoPeftModelForSeq2SeqLM

            self.model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path)
        except Exception:
            from peft import AutoPeftModelForCausalLM

            self.model = AutoPeftModelForCausalLM.from_pretrained(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Modelo fine-tunado carregado de: {model_path}")

    def generate(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 256,
    ) -> str:
        """
        Gera uma resposta usando o modelo fine-tunado.

        Args:
            question: pergunta medica
            context: contexto clinico opcional
            max_new_tokens: numero maximo de tokens na resposta

        Returns:
            resposta gerada pelo modelo
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Carregue ou treine o modelo antes de gerar respostas.")

        import torch

        prompt = (
            "### Instrucao:\nVoce e um assistente medico especializado. "
            "Responda com base no contexto e nos protocolos internos.\n\n"
            f"### Contexto:\n{context}\n\n"
            f"### Pergunta:\n{question}\n\n"
            "### Resposta:\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Para modelos causais, remove o prompt da resposta gerada
        if "### Resposta:\n" in response:
            response = response.split("### Resposta:\n")[-1].strip()

        return response
