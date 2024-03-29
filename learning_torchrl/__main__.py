import dataclasses
import time

import typer
from rich import print
from typer_config.decorators import dump_json_config, use_json_config

from learning_torchrl import Config, setup_experiment

setup_experiment()
import time

from eztils.torch import seed_everything
from eztils.typer import dataclass_option
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch.optim import Adam
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.modules import MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

from learning_torchrl import LOG_DIR, version

app = typer.Typer(
    name="learning_torchrl",
    help="project_tag",
    add_completion=False,
)


@app.command(name="")
@use_json_config()
@dump_json_config(str(LOG_DIR / "config.json"))
def main(
    conf: dataclass_option(Config) = "{}",  # type: ignore,
    wandb: bool = False,
) -> None:
    """Print a greeting with a giving name."""
    conf: Config = conf  # for type hinting
    seed_everything(conf.seed)

    print(f"[bold green]Welcome to learning_torchrl v{version}[/]")
    print(f"config {type(conf)}: {conf}")

    env = TransformedEnv(GymEnv(conf.env), StepCounter())
    env.set_seed(conf.seed)
    QValueModule(env.action_spec)

    value_mlp = MLP(
        out_features=env.action_spec.shape[-1],
        num_cells=[conf.policy_hidden] * conf.policy_layers,
    )
    value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
    policy = Seq(value_net, QValueModule(env.action_spec))
    # exploration_module = EGreedyModule(
    #     env.action_spec, annealing_num_steps=100_000, eps_init=0.5 # TODO put into conf
    # )
    # policy_explore = Seq(policy, exploration_module)

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=conf.frames_per_batch,
        total_frames=-1,
        init_random_frames=conf.init_rand_steps,
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

    loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
    optim = Adam(loss.parameters(), lr=conf.lr)
    updater = SoftUpdate(loss, eps=conf.soft_update)

    total_count = 0
    total_episodes = 0
    t0 = time.time()
    for i, data in enumerate(collector):
        # Write data in replay buffer
        rb.extend(data)
        max_length = rb[:]["next", "step_count"].max()
        if len(rb) > conf.init_rand_steps:
            # Optim loop (we do several optim steps
            # per batch collected for efficiency)
            for _ in range(conf.optim_steps):
                sample = rb.sample(conf.batch_size)
                loss_vals = loss(sample)
                loss_vals["loss"].backward()
                optim.step()
                optim.zero_grad()
                # # Update exploration factor
                # exploration_module.step(data.numel())
                # Update target params
                updater.step()
                if i % 10:
                    torchrl_logger.info(
                        f"Max num steps: {max_length}, rb length {len(rb)}"
                    )
                total_count += data.numel()
                total_episodes += data["next", "done"].sum()

        if max_length > 200:
            break

    t1 = time.time()

    torchrl_logger.info(
        f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
    )


app()
