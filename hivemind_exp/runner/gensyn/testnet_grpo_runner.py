import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import hivemind
from datasets import Dataset
from trl import GRPOConfig, ModelConfig

from hivemind_exp.chain_utils import (
    coordinator_contract,
    send_chain_txn,
    setup_account,
    setup_web3,
)
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
from hivemind_exp.trainer.gensyn.testnet_grpo_trainer import TestnetGRPOTrainer

logger = logging.getLogger(__name__)


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

    def register_peer(self, peer_id):
        logger.info(f"Registering self with peer ID: {peer_id}")
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.registerPeer(peer_id).build_transaction(
                {
                    "gas": 500000,
                    "gasPrice": self.web3.to_wei("50", "gwei"),
                }
            ),
        )

    def setup_dht(self, grpo_args):
        initial_peers = grpo_args.initial_peers
        if not initial_peers:
            logger.info("Cannot locate on-chain initial peers; running alone.")

        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        logger.info(f"🐝 Joining swarm with initial_peers = {initial_peers}")

        peer_id = str(dht.peer_id)
        self.name = self._set_animal_name(peer_id)
        self.register_peer(peer_id)
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
