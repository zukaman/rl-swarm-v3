import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Tuple

import hivemind
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, ModelConfig
from peft import LoraConfig, get_peft_model

from hivemind_exp.gsm8k.stage_utils import gsm8k_stage_data
from hivemind_exp.hivemind_utils import HivemindNode
from hivemind_exp.name_utils import get_name_from_peer_id
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer

logger = logging.getLogger(__name__)

@dataclass
class GRPOArguments:
    # Hivemind arguments
    initial_peers: list[str] = field(default_factory=list)
    public_maddr: str | None = None
    host_maddr: str | None = None
    identity_path: str | None = None
    max_rounds: int = 100

    # Model arguments
    dataset_id_or_path: str = "openai/gsm8k"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str | None = None
    number_of_data_samples: int = 50000
    public_maddr: str | None = None

    # LoRA arguments
    # Используем другие имена для параметров LoRA, чтобы избежать конфликтов с автогенерацией аргументов
    peft_enable: bool = False  # Вместо use_lora
    peft_rank: int = 16        # Вместо lora_rank
    peft_alpha: int = 32       # Вместо lora_alpha
    peft_dropout: float = 0.05 # Вместо lora_dropout
    peft_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])  # Вместо target_modules

    #Hugging Face Hub arguments
    hf_token: str | None = None


class GRPORunner:
    def get_model(self, args: GRPOConfig, model_name: str, script_args: GRPOArguments = None):
        model_init_kwargs = args.model_init_kwargs or {}
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_init_kwargs)
        
        # Apply LoRA if enabled in script_args
        if script_args and script_args.peft_enable:
            logger.info("=" * 50)
            logger.info("APPLYING LORA FINE-TUNING")
            logger.info(f"LoRA rank: {script_args.peft_rank}")
            logger.info(f"LoRA alpha: {script_args.peft_alpha}")
            logger.info(f"LoRA dropout: {script_args.peft_dropout}")
            logger.info(f"LoRA target modules: {script_args.peft_modules}")
            logger.info("=" * 50)
            
            # Count total parameters before LoRA
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters before LoRA: {total_params:,}")
            
            lora_config = LoraConfig(
                r=script_args.peft_rank,
                lora_alpha=script_args.peft_alpha,
                lora_dropout=script_args.peft_dropout,
                target_modules=script_args.peft_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Log detailed LoRA info
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total trainable parameters with LoRA: {trainable_params:,}")
            logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")
            
            # Print detailed trainable parameters information
            model.print_trainable_parameters()
            
            # Log LoRA adapter names for verification
            logger.info("LoRA adapter names:")
            for name, _ in model.named_modules():
                if 'lora' in name.lower():
                    logger.info(f"  - {name}")
            
            logger.info("=" * 50)
        
        return model

    def get_tokenizer_name(self, model_args: ModelConfig, script_args: GRPOArguments):
        if script_args.tokenizer_name_or_path:
            return script_args.tokenizer_name_or_path
        if model_args.model_name_or_path:
            return model_args.model_name_or_path
        raise ValueError("unable to resolve tokenizer name")

    def _dht_kwargs(self, grpo_args):
        kwargs = {}
        initial_peers = grpo_args.initial_peers
        if initial_peers:
            kwargs["initial_peers"] = initial_peers

        if public_maddr := grpo_args.public_maddr:
            kwargs["announce_maddrs"] = [public_maddr]

        if host_maddr := grpo_args.host_maddr:
            kwargs["host_maddrs"] = [host_maddr]

        if identity_path := grpo_args.identity_path:
            kwargs["identity_path"] = identity_path

        return kwargs

    def _get_animal_name(self, peer_id):
        animal_name = get_name_from_peer_id(peer_id)
        logger.info(f"🐱 Hello 🐈 [{animal_name}] 🦮 [{peer_id}]!")
        return animal_name

    def setup_dht(self, grpo_args):
        initial_peers = grpo_args.initial_peers
        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        if initial_peers:
            logger.info(f"🐝 Joining swarm with initial_peers = {initial_peers}")
        else:
            first_visible = str(dht.get_visible_maddrs()[0])
            logger.info(f"🤖 Starting swarm at {first_visible}")

        self.name = self._get_animal_name(str(dht.peer_id))
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
        trainer_factory_fn: Callable = HivemindGRPOTrainer,
    ):
        #########################
        # Log parameters
        #########################
        logger.debug(f"Model parameters {model_args}")
        logger.debug(f"Training/evaluation parameters {training_args}")

        batch_size = 2
        training_args.per_device_train_batch_size = batch_size
        training_args.num_generations = batch_size

        ############################
        # Log into HF hub if wanted
        ############################
        if (grpo_args.hf_token not in [None, "None"]):
            training_args.push_to_hub_token = grpo_args.hf_token
            login(token=training_args.push_to_hub_token, add_to_git_credential=True)
        else:
            training_args.push_to_hub_token = None

        ################
        # Load tokenizer
        ################
        tokenizer = AutoTokenizer.from_pretrained(
            self.get_tokenizer_name(model_args, grpo_args),
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        #########################
        # Create DHT via Hivemind
        #########################
        dht = self.setup_dht(grpo_args)

        #####################################
        # Load datasets, prepare, and format
        #####################################
        train_dataset, test_dataset = initial_datasets_fn()

        #########################
        # Instantiate DPO trainer
        #########################
        model_name_or_path = model_args.model_name_or_path
        assert model_name_or_path
        model = self.get_model(training_args, model_name_or_path, grpo_args)

        initial_peers = grpo_args.initial_peers
        if initial_peers:
            node = HivemindNode(model_name_or_path, str(dht.peer_id))
        else:
            node = HivemindNode.coordinator(model_name_or_path, str(dht.peer_id))

        stage_data = gsm8k_stage_data(dht, node, train_dataset, test_dataset)
        stage_data.max_rounds = grpo_args.max_rounds
        trainer = trainer_factory_fn(
            dht=dht,
            node=node,
            model=model,
            tokenizer=tokenizer,
            config=training_args,
            stage_data=stage_data,
            log_tag=self.name,
        )

        ###############
        # Training loop
        ###############
        logger.info(
            f"Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs"
        )
        trainer.train()
