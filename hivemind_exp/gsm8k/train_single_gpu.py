
import logging

import colorlog
from trl import GRPOConfig, ModelConfig, TrlParser

from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples
from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
    TestnetGRPOArguments,
    TestnetGRPORunner,
)
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner


def main():
    # Setup logging.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(light_red)s%(levelname)s:%(name)s:%(message)s'))
    root_logger.addHandler(handler)

    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig)) # type: ignore
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()

    # Run main training loop.
    if testnet_args.wallet_private_key:
        runner = TestnetGRPORunner(testnet_args)
    else:
        runner = GRPORunner()

    runner.run(model_args, grpo_args, training_args, get_stage1_samples)

if __name__ == "__main__":
    main()
