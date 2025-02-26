from functools import partial
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import hivemind
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from datasets import Dataset

from hivemind_exp.dht_utils import *
from hivemind_exp.gsm8k.stage_merger import *
from hivemind_exp.gsm8k.stage_utils import gsm8k_stage_data
from hivemind_exp.tests.fake_data import *
from hivemind_exp.trainer.hivemind_grpo_trainer import (
    HivemindGRPOTrainer,
    get_dht_value,
)
from hivemind_exp.utils import SingleStageData

TEST_MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def get_model_config(tmp_path):
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    config = GRPOConfig(
        output_dir=tmp_path,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        max_steps=1,
    )
    return model, config


def wrap_datasets_fn(stage: SingleStageData, check):
    orig = stage.datasets_fn

    def wrapped(r, s):
        value = orig(r, s)
        check(value[0])
        return value

    stage.datasets_fn = wrapped


def check_dataset(prefix: str, min_count: int, dataset: Dataset):
    c = 0
    for feature in dataset.features:
        if feature.startswith(prefix):
            c += 1
    assert c >= min_count


def create_dht_and_trainer(tmp_path, node, min_peers=1, initial_peers=[]):
    dht = hivemind.DHT(start=True, initial_peers=initial_peers)
    model, config = get_model_config(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

    # Always check stage merging.

    def check_merged_stage1_dataset(dataset: Dataset):
        # print(f"Merged stage 1 for: {node.uuid}", dataset)
        check_dataset("agent_answers", min_peers, dataset)

    def check_merged_stage2_dataset(dataset: Dataset):
        # print(f"Merged stage 2 for: {node.uuid}", dataset)
        check_dataset("agent_opinion", min_peers, dataset)

    stage_data = gsm8k_stage_data(dht, node, SAMPLES, SAMPLES)
    stage_data.max_rounds = 1
    stage_data.stages[0].datasets_fn = lambda r, s: (SAMPLES, SAMPLES)  # type: ignore
    wrap_datasets_fn(stage_data.stages[1], check_merged_stage1_dataset)
    wrap_datasets_fn(stage_data.stages[2], check_merged_stage2_dataset)

    trainer = HivemindGRPOTrainer(
        dht=dht,
        node=node,
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage_data=stage_data,
    )
    return dht, trainer


def test_gsm8k_stage_data(tmp_path):
    coord = HivemindNode.coordinator("test")
    nodes = [HivemindNode("test") for _ in range(3)]

    dht_trainers = [create_dht_and_trainer(Path(tmp_path) / "C", coord, min_peers=2)]
    dht0 = dht_trainers[0][0]
    for i, node in enumerate(nodes):
        dht_trainers.append(
            create_dht_and_trainer(
                Path(tmp_path) / str(i),
                node,
                min_peers=2,
                initial_peers=dht0.get_visible_maddrs(),
            )
        )

    for dht, _ in dht_trainers:
        _ = dht.get_visible_maddrs(latest=True)

    with ThreadPoolExecutor() as executor:
        for dht, trainer in dht_trainers:
            executor.submit(trainer.train)

    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (0, 2)  # 1 round, 3 stages

    def check_outputs(outputs: dict[str, tuple] | None, output_checks={}):
        assert outputs
        qo = outputs[QUESTION][1]
        assert qo["question"] == QUESTION
        assert qo["answer"] == "42"
        for k, check in output_checks.items():
            assert k in qo
            assert check(qo[k])

    for r, s in itertools.product(range(1), range(3)):
        match s:
            case 0:
                checks = {}
            case 1:
                # Only one before merging.
                checks = {"agent_opinion": lambda c: len(c) == 1}
            case 2:
                checks = {"final_agent_decision": lambda c: len(c) == 1}
            case _:
                checks = {}

        for i in range(len(nodes)):
            check_outputs(
                get_dht_value(dht0, key=outputs_key(nodes[i].uuid, r, s), latest=True),
                checks,
            )

        rewards = get_dht_value(dht0, key=rewards_key(r, s), latest=True)
        assert rewards
        assert rewards.keys() == set([CK] + [node.uuid for node in nodes])


def test_gsm8k_delayed_join(tmp_path):
    node0 = HivemindNode.coordinator("test")
    node1 = HivemindNode("test")

    dht0, trainer0 = create_dht_and_trainer(Path(tmp_path) / "0", node0)
    dht1, trainer1 = create_dht_and_trainer(
        Path(tmp_path) / "1",
        node1,
        initial_peers=dht0.get_visible_maddrs(),
    )
    _ = dht0.get_visible_maddrs(latest=True)
    _ = dht1.get_visible_maddrs(latest=True)

    def delayed_train():
        while trainer0.node.stage_num == 0:
            time.sleep(0.5)

        trainer1.train()

    with ThreadPoolExecutor() as executor:
        executor.submit(trainer0.train)
        executor.submit(delayed_train)

    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (0, 2)  # 1 round, 3 stages

    for r, s in itertools.product(range(1), range(3)):
        outputs0 = get_dht_value(dht1, key=outputs_key(node0.uuid, r, s), latest=True)
        assert outputs0

        if s > 0:
            outputs1 = get_dht_value(
                dht0, key=outputs_key(node1.uuid, r, s), latest=True
            )
            assert outputs1
