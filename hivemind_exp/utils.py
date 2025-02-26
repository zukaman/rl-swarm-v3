import torch

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TypeVar
import uuid

COORDINATOR_KEY = "GENSYN"

@dataclass
class HivemindNode:
    # Node metadata.
    model_name: str
    node_name: str = "node"
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Q&A outputs from the last training step.
    outputs: Mapping[Any, Any] = field(default_factory=dict)

    # Reward outputs from the last training.
    rewards: Sequence[float | int] = field(default_factory=list)

    # Values incremented by coordinator.
    round_num: int = 0
    stage_num: int = 0

    out_expiration: int = 60 * 60 * 4 # hours

    def is_coordinator(self) -> bool:
        return self.uuid == COORDINATOR_KEY

    @staticmethod
    def coordinator(model_name: str):
        return HivemindNode(model_name, uuid=COORDINATOR_KEY)


# Takes round + stage.
DatasetsFn = Callable[
    [int, int], tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
]

MergeFn = Callable[[list], dict[str, dict]]
LossFn = Callable[[list], dict[str, float]]


@dataclass
class SingleStageData:
    name: str
    reward_funcs: list[Callable]
    datasets_fn: DatasetsFn  # For train / test datasets.


@dataclass
class StageData:
    stages: Sequence[SingleStageData]
    max_rounds: int = 100
    train_timeout: int = 60 * 60 * 24 * 4 # days
    round_timeout: int = 60 * 60 * 4 # hours

    def __len__(self):
        return len(self.stages)
