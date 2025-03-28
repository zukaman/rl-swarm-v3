import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import hivemind
from datasets import Dataset
from trl import GRPOConfig, ModelConfig

from hivemind_exp.gensyn import coordinator_contract, setup_account, setup_web3
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
from hivemind_exp.trainer.gensyn.testnet_grpo_trainer import TestnetGRPOTrainer

logger = logging.getLogger(__name__)


########################
# Custom dataclasses
########################
@dataclass
class TestnetGRPOArguments:
    wallet_private_key: str | None = None  # EOA wallet private key


class TestnetGRPORunner(GRPORunner):
    def __init__(self, args: TestnetGRPOArguments) -> None:
        self.web3 = setup_web3()
        self.account = setup_account(self.web3, args.wallet_private_key)
        self.contract = coordinator_contract(self.web3)

    def get_initial_peers(self) -> list[str]:
        return self.contract.functions.getBootnodes().call()

    def setup_dht(self, grpo_args):
        initial_peers = grpo_args.initial_peers
        if not initial_peers:
            raise ValueError("Cannot locate on-chain initial peers. Exiting.")

        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        logger.info(f"Joining swarm with initial_peers = {initial_peers}")
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
    ):
        initial_peers = self.get_initial_peers()
        logger.info(f"Retrieved initial peers from chain: {initial_peers}")
        grpo_args.initial_peers = initial_peers
        super().run(
            model_args,
            grpo_args,
            training_args,
            initial_datasets_fn,
            partial(
                TestnetGRPOTrainer,
                web3=self.web3,
                account=self.account,
                contract=self.contract,
            ),
        )
