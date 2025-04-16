import gc
import hashlib
import logging
import time
import traceback
from typing import Any

import datasets
import torch
from hivemind.dht import DHT
from hivemind.utils import get_dht_time
from trl import GRPOConfig, GRPOTrainer

from hivemind_exp.debug_utils import print_system_info
from hivemind_exp.dht_utils import (
    ROUND_STAGE_NUMBER_KEY,
    get_dht_value,
    get_round_and_stage,
    leaderboard_key,
    node_outputs_key,
    rewards_key,
)
from hivemind_exp.hivemind_utils import HivemindNode, StageData
from hivemind_exp.name_utils import get_name_from_peer_id


MAX_TRAIN_FAILS = 5
CADENCE_OF_UPDATE_STEPS = 4


class HivemindGRPOTrainer:
    """
    Subclass of GRPOTrainer that implements multi-stage GRPO by publishing
    intermediate results to a connected Hivemind DHT.
    """

    class PublishingGRPOTrainer(GRPOTrainer):
        def __init__(
            self,
            node: HivemindNode,
            dht: DHT,
            tokenizer,
            logger,
            **kwargs,
        ):
            self.node = node
            self.dht = dht
            self.logger = logger
            self.stage_rewards = 0.0
            
            # Log if we're using a PEFT/LoRA model
            model = kwargs.get("model")
            if model and hasattr(model, "is_peft_model") and model.is_peft_model:
                self.logger.info("=" * 50)
                self.logger.info("INITIALIZING TRAINER WITH LORA MODEL")
                self.logger.info(f"Model type: {type(model).__name__}")
                if hasattr(model, "active_adapter"):
                    self.logger.info(f"Active adapter: {model.active_adapter}")
                if hasattr(model, "modules_to_save"):
                    self.logger.info(f"Modules to save: {model.modules_to_save}")
                self.logger.info("=" * 50)
            
            super().__init__(processing_class=tokenizer, **kwargs)

        def publish_leaderboard(self):
            r, s = self.node.round_num, self.node.stage_num
            curr_rewards: dict[str, Any] | None = get_dht_value(
                self.dht, key=rewards_key(r, s), latest=True
            )
            if curr_rewards:
                # Sorted list of (node_key, reward) pairs.
                leaderboard = list(
                    sorted(
                        curr_rewards.items(), key=lambda t: (t[1], t[0]), reverse=True
                    )
                )
                self.dht.store(
                    key=leaderboard_key(r, s),
                    value=leaderboard,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
            else:
                self.logger.info(f"Can't retrieve round {r} stage {s - 1} rewards")

        # Store initial LoRA weights for comparison
        initial_lora_weights = {}
        
        def compute_loss(self, model, inputs, *args, **kwargs):
            # First time initialization of weight tracking
            if hasattr(model, "is_peft_model") and model.is_peft_model and not hasattr(self, '_lora_weight_tracking_initialized'):
                self._lora_weight_tracking_initialized = True
                
                # Store initial LoRA weights for tracking changes
                for name, param in model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        # Store a clone of the initial weights for comparison
                        if hasattr(param, 'data'):
                            self.initial_lora_weights[name] = param.data.clone().detach()
                
                self.logger.info(f"Initialized LoRA weight tracking for {len(self.initial_lora_weights)} parameters")
            
            # Log LoRA usage periodically during training
            if hasattr(model, "is_peft_model") and model.is_peft_model and self.state.global_step % 50 == 0:
                self.logger.info(f"Step {self.state.global_step}: Using LoRA model for training")
                
                # Check if any gradients are flowing through LoRA layers
                if self.state.global_step % 200 == 0:  # Less frequent check to avoid log spam
                    has_grad = False
                    for name, param in model.named_parameters():
                        if 'lora' in name.lower() and param.requires_grad:
                            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                                has_grad = True
                                self.logger.info(f"LoRA parameter {name} has non-zero gradients")
                                break
                    if not has_grad:
                        self.logger.warning("No gradients detected in LoRA parameters! Check your configuration.")
                
                # Log weight changes for LoRA parameters occasionally
                if self.state.global_step % 300 == 0 and hasattr(self, '_lora_weight_tracking_initialized'):
                    self.logger.info("=" * 30)
                    self.logger.info(f"LORA WEIGHT CHANGES AT STEP {self.state.global_step}")
                    checked = 0
                    
                    for name, param in model.named_parameters():
                        if name in self.initial_lora_weights and checked < 3:
                            # Calculate change from initial weights
                            initial = self.initial_lora_weights[name]
                            current = param.data
                            
                            if initial.shape == current.shape:
                                # Calculate statistics about the changes
                                abs_diff = torch.abs(current - initial)
                                mean_change = abs_diff.mean().item()
                                max_change = abs_diff.max().item()
                                
                                # Calculate percentage of weights that changed significantly
                                significant_change_threshold = 1e-6
                                pct_changed = (abs_diff > significant_change_threshold).float().mean().item() * 100
                                
                                self.logger.info(f"Parameter: {name}")
                                self.logger.info(f"  - Mean absolute change: {mean_change:.8f}")
                                self.logger.info(f"  - Max absolute change: {max_change:.8f}")
                                self.logger.info(f"  - Percentage weights changed: {pct_changed:.2f}%")
                                
                                checked += 1
                    
                    self.logger.info("=" * 30)
            
            loss = super().compute_loss(model, inputs, *args, **kwargs)
            # Reward function must save node.outputs + node.rewards!
            # This is only here to publish to the DHT at the right time.
            # Only publish to DHT every N steps
            if self.state.global_step % CADENCE_OF_UPDATE_STEPS == 0:
                question = self.node.outputs["question"]
                q_hash = hashlib.md5(question.encode()).hexdigest()

                value = (time.time(), self.node.outputs)
                self.dht.store(
                    key=node_outputs_key(self.node),
                    subkey=q_hash,
                    value=value,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
                self.node.put_stage_outputs(
                    self.node.round_num, self.node.stage_num, q_hash, value
                )

                # Just the latest.
                self.stage_rewards += sum(self.node.rewards)
                self.dht.store(
                    key=rewards_key(self.node.round_num, self.node.stage_num),
                    subkey=self.node.key,
                    value=self.stage_rewards,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
            if self.node.is_coordinator:
                self.publish_leaderboard()

            return loss

    def __init__(
        self,
        node: HivemindNode,
        dht: DHT,
        stage_data: StageData,
        config: GRPOConfig,
        model,
        tokenizer,
        log_tag=None,
        **kwargs,
    ):
        # The single coordinator is responsible for incrementing round + stage numbers.
        # TODO(lou): Allow ability to choose different coordinators?
        self.node = node
        self.dht = dht

        self.stage_data = stage_data

        self.config = config
        self.config.dataloader_num_workers=0  # Default: 8+
        assert self.config.output_dir
        self.config.output_dir += f"-{get_name_from_peer_id(self.node.key, True)}"  # TODO: Add animal name to save path in more appropriate spot
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not log_tag:
            log_tag = self.node.key

        self.logger = logging.getLogger(f"{__name__}:{log_tag}")

    def wait_for(self, result_fn=lambda: None, interval=10, timeout=30):
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            result = result_fn()
            if result is None:
                time.sleep(interval)
            else:
                break

        return result

    def train_stages(self, round_num, start_stage, is_coordinator):
        # TODO: Needs checkpoint loading
        self.node.round_num = round_num
        for i, stage in enumerate(self.stage_data.stages[start_stage:]):
            stage_num = start_stage + i
            self.node.stage_num = stage_num

            if is_coordinator:
                self.dht.store(
                    key=ROUND_STAGE_NUMBER_KEY,
                    value=(self.node.round_num, stage_num),
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )

            self.logger.info(f"📈 Training round: {round_num} stage: {stage_num}")
            
            # Log LoRA status at the beginning of each stage
            if hasattr(self.model, "is_peft_model") and self.model.is_peft_model:
                self.logger.info("=" * 50)
                self.logger.info(f"LORA STATUS AT START OF ROUND {round_num} STAGE {stage_num}")
                
                # Log adapter information
                if hasattr(self.model, "active_adapter"):
                    self.logger.info(f"Active adapter: {self.model.active_adapter}")
                
                # Count trainable parameters
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
                
                # Sample some initial LoRA weights
                self.logger.info("Initial LoRA weight samples:")
                checked = 0
                for name, param in self.model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        if checked < 5:  # Limit to avoid log spam
                            mean = param.data.mean().item()
                            std = param.data.std().item()
                            self.logger.info(f"  - {name}: mean={mean:.6f}, std={std:.6f}")
                            checked += 1
                self.logger.info("=" * 50)
            
            train_dataset, test_dataset = stage.datasets_fn(round_num, stage_num)
            kwargs = {
                "model": self.model,
                "args": self.config,
                "reward_funcs": stage.reward_funcs,
                "train_dataset": train_dataset,
                "eval_dataset": test_dataset,
            }
            trainer = HivemindGRPOTrainer.PublishingGRPOTrainer(
                self.node, self.dht, self.tokenizer, self.logger, **kwargs
            )
            self.train_and_save(trainer, train_dataset)
            
            # Print LoRA summary after training
            if hasattr(self.model, "is_peft_model") and self.model.is_peft_model:
                self.logger.info("=" * 50)
                self.logger.info(f"LORA TRAINING SUMMARY FOR ROUND {round_num} STAGE {stage_num}")
                
                # Count and report parameter changes
                total_lora_params = 0
                total_changed_params = 0
                significant_threshold = 1e-5
                
                for name, param in self.model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        param_count = param.numel()
                        total_lora_params += param_count
                        
                        # If we have initial weights for comparison
                        if hasattr(trainer, 'initial_lora_weights') and name in trainer.initial_lora_weights:
                            initial = trainer.initial_lora_weights[name]
                            current = param.data
                            
                            if initial.shape == current.shape:
                                # Count significantly changed weights
                                changed = (torch.abs(current - initial) > significant_threshold).sum().item()
                                total_changed_params += changed
                
                # Report overall statistics
                if total_lora_params > 0:
                    change_percentage = (total_changed_params / total_lora_params) * 100
                    self.logger.info(f"Total LoRA parameters: {total_lora_params:,}")
                    self.logger.info(f"Parameters changed significantly: {total_changed_params:,} ({change_percentage:.2f}%)")
                
                self.logger.info("=" * 50)
            
            self.logger.info(
                f"📉 Finished training round: {round_num} stage: {stage_num}"
            )

        # Push to HF hub if desired
        # TODO: Come back and add additional logic checking if they've provided access token+HF username
        if self.config.push_to_hub_token is not None:
            self.logger.info("Pushing model to Hugging Face Hub...")
            try:
                trainer.push_to_hub(
                    tags=[
                        "rl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {get_name_from_peer_id(self.node.key)}",
                    ]
                )
                time.sleep(1)
            except Exception:
                self.logger.info(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading"
                )

        self.cleanup()

    def cleanup(self):
        # Clear various stage caches.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():  # type: ignore
            torch.mps.empty_cache()  # type: ignore
        try:
            if torch.xpu.is_available():  # type: ignore
                torch.xpu.empty_cache()  # type: ignore
        except AttributeError:
            pass

        self.node.clear_stage_cache()

    def train_and_save(self, trainer, train_dataset):
        # Print detailed model info before training
        if hasattr(trainer.model, "is_peft_model") and trainer.model.is_peft_model:
            self.logger.info("=" * 50)
            self.logger.info("LORA MODEL STRUCTURE BEFORE TRAINING")
            # Print model structure focusing on LoRA modules
            for name, module in trainer.model.named_modules():
                if 'lora' in name.lower():
                    self.logger.info(f"Found LoRA module: {name} - {type(module).__name__}")
                    if hasattr(module, 'weight'):
                        self.logger.info(f"  - Shape: {module.weight.shape if hasattr(module.weight, 'shape') else 'N/A'}")
                    if hasattr(module, 'r'):
                        self.logger.info(f"  - Rank: {module.r}")
                    if hasattr(module, 'lora_alpha'):
                        self.logger.info(f"  - Alpha: {module.lora_alpha}")
            
            # Verify LoRA weights exist and require gradients
            lora_layers = 0
            trainable_params = 0
            for name, param in trainer.model.named_parameters():
                if 'lora' in name.lower():
                    lora_layers += 1
                    if param.requires_grad:
                        trainable_params += param.numel()
                        self.logger.info(f"  - Trainable LoRA param: {name}, shape: {param.shape}")
            
            self.logger.info(f"Total LoRA layers found: {lora_layers}")
            self.logger.info(f"Total trainable parameters: {trainable_params:,}")
            self.logger.info("=" * 50)
            
        # Regular training loop
        for num_fails in range(MAX_TRAIN_FAILS):
            try:
                train_result = trainer.train()
                break
            except (BlockingIOError, EOFError) as e:
                self.logger.warning(f"DHT IPC error: {e}. Restarting training...")
                self.cleanup()  # Clear GPU/caches
                time.sleep(5)
                continue

        # Log and save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        self.logger.info("Saving model")
        trainer.model.config.use_cache = True
        
        # Check if the model is using PEFT/LoRA
        if hasattr(trainer.model, "is_peft_model") and trainer.model.is_peft_model:
            self.logger.info("=" * 50)
            self.logger.info("SAVING PEFT/LORA MODEL ADAPTER WEIGHTS")
            self.logger.info(f"Model checkpoint directory: {self.config.output_dir}")
            
            # Log active adapters
            if hasattr(trainer.model, "active_adapter"):
                self.logger.info(f"Active adapter: {trainer.model.active_adapter}")
            if hasattr(trainer.model, "peft_config"):
                self.logger.info("PEFT configuration:")
                for adapter_name, config in trainer.model.peft_config.items():
                    self.logger.info(f"Adapter: {adapter_name}")
                    for k, v in config.to_dict().items():
                        self.logger.info(f"  - {k}: {v}")
            
            # Sample and log some LoRA weights to verify they've changed
            self.logger.info("Sample of trained LoRA weights:")
            checked_weights = 0
            for name, param in trainer.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    if checked_weights < 5:  # Limit to 5 weights to avoid log spam
                        # Get some statistics about the weights
                        if param.numel() > 0:
                            mean = param.data.mean().item()
                            std = param.data.std().item()
                            min_val = param.data.min().item()
                            max_val = param.data.max().item()
                            self.logger.info(f"  - {name}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")
                            checked_weights += 1
            
            # Save the adapter weights
            trainer.model.save_pretrained(self.config.output_dir)
            self.logger.info("LoRA adapter weights saved successfully")
            self.logger.info("=" * 50)
        else:
            # Save full model weights for non-LoRA models
            self.logger.info("Saving full model weights (LoRA not detected)")
            trainer.save_model(self.config.output_dir)
            
        self.logger.info(f"Model saved to {self.config.output_dir}")
        assert self.config.distributed_state
        self.config.distributed_state.wait_for_everyone()  # wait for all processes to load

        self.tokenizer.save_pretrained(self.config.output_dir)
        self.logger.info(f"Tokenizer saved to {self.config.output_dir}")

    def get_round_and_stage(self):
        return get_round_and_stage(self.dht)

    def coordinator_train(self):
        round_num = 0
        start_time = time.monotonic()
        while (
            round_num < self.stage_data.max_rounds
            and time.monotonic() - start_time < self.stage_data.train_timeout
        ):
            self.logger.info(f"🤖 Starting new round: {round_num}")

            _ = self.dht.get_visible_maddrs(latest=True)
            self.train_stages(round_num, 0, is_coordinator=True)

            round_num += 1
            if round_num == self.stage_data.max_rounds:
                return

        self.logger.info("Training timed out!")

    def follower_train(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 5
    ):
        done_rounds = set()
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = (
            check_interval  # Exponential backoff for already finished rounds.
        )
        while time.monotonic() - start_time < self.stage_data.train_timeout:
            curr_time = time.monotonic()
            _ = self.dht.get_visible_maddrs(latest=True)

            # Retrieve current round and stage.
            try:
                round_num, stage = self.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    self.logger.debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num not in done_rounds:
                self.logger.info(
                    f"🐝 Joining round: {round_num} starting at stage: {stage}"
                )
                try:
                    self.train_stages(round_num, stage, is_coordinator=False)
                except datasets.exceptions.DatasetGenerationError:
                    if stage > 0:
                        self.logger.info("Re-attempting training starting at stage 0!")

                        # Start over from stage 0.
                        self.train_stages(round_num, 0, is_coordinator=False)
                    else:
                        raise

                done_rounds.add(round_num)
                check_backoff = check_interval  # Reset backoff after successful round
            else:
                self.logger.info(
                    f"Already finished round: {round_num}. Next check in {check_backoff}s."
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.stage_data.max_rounds - 1:
                return

        self.logger.info("Training timed out!")

    def _train(self):
        if self.node.is_coordinator:
            self.coordinator_train()
        else:
            self.follower_train()

    def train(self):
        try:
            self._train()

        except Exception:
            self.logger.error("Encountered error during training!")
            print_system_info()
            traceback.print_exc()
            raise
