import logging
from typing import Sequence

from eth_account import Account
from web3 import Web3

from hivemind_exp.gensyn import send_chain_txn
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer

logger = logging.getLogger(__name__)


class TestnetGRPOTrainer(HivemindGRPOTrainer):
    def __init__(self, web3: Web3, account: Account, contract, **kwargs) -> None:
        self.web3 = web3
        self.account = account
        self.contract = contract
        super().__init__(**kwargs)
        self.register_peer()

    def register_peer(self):
        logger.info(f"Registering self with peer ID: {self.node.uuid}")
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.registerPeer(
                self.node.uuid
            ).build_transaction(
                {
                    "gas": 500000,
                    "gasPrice": self.web3.to_wei("50", "gwei"),
                }
            ),
        )

    def submit_winners(self, round_num: int, winners: Sequence[str]):
        logger.info(f"Submitting winners for round {round_num}: {winners}")
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.submitWinners(
                round_num, winners
            ).build_transaction(
                {
                    "gas": 500000,
                    "gasPrice": self.web3.to_wei("50", "gwei"),
                }
            ),
        )

    def get_round_and_stage(self):
        with self.web3.batch_requests() as batch:
            batch.add(self.contract.functions.currentRound())
            batch.add(self.contract.functions.currentStage())
            resp = batch.execute()

        return resp[0], resp[1]

    def train_stages(self, round_num, start_stage, is_coordinator):
        super().train_stages(round_num, start_stage, is_coordinator)
        self.submit_winners(round_num, self.stage_data.round_winner_fn())

    def train(self):
        try:
            self.follower_train()

        except Exception:
            import traceback

            traceback.print_exc()
