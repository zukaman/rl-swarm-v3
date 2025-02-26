from collections import defaultdict
from hivemind_exp.dht_utils import *
import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
import hivemind_exp.gsm8k.stage2_rewards as stage2_rewards
import hivemind_exp.gsm8k.stage3_rewards as stage3_rewards
from hivemind_exp.gsm8k.generate_prompts import *
from hivemind_exp.gsm8k.stage_merger import *
from hivemind_exp.utils import SingleStageData, StageData


def gsm8k_stage_data(dht, node, initial_train_dataset, initial_test_dataset):
    def cumulative_reward_0(**kwargs):
        return stage1_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_1(**kwargs):
        return stage2_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_2(**kwargs):
        return stage3_rewards.hivemind_cumulative_reward(node, **kwargs)

    def stage_datasets_fn(r, s, merge_fn, samples_fn):
        prev_rewards: dict[str, Any] | None = get_dht_value(
            dht, key=rewards_key(r, s - 1), latest=True
        )
        assert prev_rewards

        prev_outputs: dict[str, list] = defaultdict(list)
        for node_uuid in prev_rewards:
            prev_node_outputs: dict[str, tuple[float, dict]] | None = get_dht_value(
                dht, key=outputs_key(node_uuid, r, s - 1), latest=True
            )
            assert prev_node_outputs
            for _, outputs in prev_node_outputs.values():
                prev_outputs[node_uuid].append(outputs)

        q_to_keyed_outputs: dict[str, dict[str, Any]] = defaultdict(dict)
        for node_uuid, all_outputs in prev_outputs.items():
            for outputs in all_outputs:
                q_to_keyed_outputs[outputs["question"]][node_uuid] = outputs

        merged_qs = []
        for outputs in q_to_keyed_outputs.values():
            merged = merge_fn(outputs)
            merged_qs.append(merged)

        return samples_fn(merged_qs)

    def stage2_datasets_fn(r, s):
        return stage_datasets_fn(r, s, merge_stage1_question, get_stage2_samples)

    def stage3_datasets_fn(r, s):
        return stage_datasets_fn(r, s, merge_stage2_question, get_stage3_samples)

    return StageData(
        max_rounds=100,  # note, this gets overridden from the config file
        stages=[
            SingleStageData(
                name="0",
                reward_funcs=[
                    stage1_rewards.xmlcount_reward_func,
                    stage1_rewards.soft_format_reward_func,
                    stage1_rewards.strict_format_reward_func,
                    stage1_rewards.int_reward_func,
                    stage1_rewards.correctness_reward_func,
                    cumulative_reward_0,
                ],
                datasets_fn=lambda r, s: (initial_train_dataset, initial_test_dataset),  # type: ignore
            ),
            SingleStageData(
                name="1",
                reward_funcs=[
                    stage2_rewards.proper_id_reward_func,
                    stage2_rewards.correctness_reward_func,
                    stage2_rewards.strict_format_reward_func,
                    stage2_rewards.soft_format_reward_func,
                    stage2_rewards.xmlcount_reward_func,
                    cumulative_reward_1,
                ],
                datasets_fn=stage2_datasets_fn,  # type: ignore
            ),
            SingleStageData(
                name="2",
                reward_funcs=[
                    stage3_rewards.consensus_reward_func,
                    stage3_rewards.concensus_correctness_reward_func,
                    stage3_rewards.question_recreation_reward_func,
                    stage3_rewards.final_correctness_reward_func,
                    stage3_rewards.strict_format_reward_func,
                    stage3_rewards.soft_format_reward_func,
                    stage3_rewards.xmlcount_reward_func,
                    cumulative_reward_2,
                ],
                datasets_fn=stage3_datasets_fn,  # type: ignore
            ),
        ],
    )
