from dataclasses import dataclass

import ml.api as ml
from torch import Tensor
from torch.distributions.normal import Normal

from project.models.a2c_discrete import A2CDiscreteModel
from project.tasks.environments.binary import Action, Environment, State


@dataclass
class BinaryTaskConfig(ml.ReinforcementLearningTaskConfig):
    gamma: float = ml.conf_field(0.99, help="The discount factor")
    gae_lmda: float = ml.conf_field(0.9, help="The GAE factor (higher means more variance, lower means more bias)")
    clip: float = ml.conf_field(0.16, help="The PPO clip factor")
    val_coef: float = ml.conf_field(0.5, help="The value loss coefficient")
    ent_coef: float = ml.conf_field(1e-2, help="The entropy coefficient")
    sample_clip_interval: int = ml.conf_field(25, help="Sample a clip with this frequency")
    normalize_advantages: bool = ml.conf_field(False, help="If set, normalize advantages")


Model = A2CDiscreteModel
Output = tuple[Tensor, Normal]
Loss = dict[str, Tensor]


@ml.register_task("binary", BinaryTaskConfig)
class BinaryTask(ml.ReinforcementLearningTask[ BinaryTaskConfig, Model, State, Action, Output, Loss]):
    def __init__(self, config: BinaryTaskConfig):
        super().__init__(config)

    def get_actions(self, model: Model, states: list[State], optimal: bool) -> list[Action]:
        collated_states = self._device.recursive_apply(self.collate_fn(states))
        breakpoint()
        board = collated_states.board.flatten(-2)
        value = model.forward_value_net(board).cpu()
        p_dist = model.forward_policy_net(collated_states.board)
        action = p_dist.mode if optimal else p_dist.sample()
        breakpoint()
        log_prob, action = p_dist.log_prob(action).cpu(), action.cpu()
        return [Action.from_policy(c, p, v) for c, p, v in zip(action.unbind(0), log_prob.unbind(0), value.unbind(0))]

    def get_environment(self) -> Environment:
        return Environment()

    def run_model(self, model: Model, batch: tuple[State, Action], state: ml.State) -> Output:
        raise NotImplementedError

    def compute_loss(self, model: Model, batch: tuple[State, Action], state: ml.State, output: Output) -> Loss:
        raise NotImplementedError


def run_adhoc_test() -> None:
    """Runs adhoc tests for this task.

    Usage:
        python -m project.tasks.binary
    """

    ml.configure_logging(use_tqdm=True)
    config = BinaryTaskConfig()
    config.environment.max_steps = 100
    task = BinaryTask(config)
    task.sample_clip(save_path="out/binary.mp4", writer="opencv")


if __name__ == "__main__":
    run_adhoc_test()
