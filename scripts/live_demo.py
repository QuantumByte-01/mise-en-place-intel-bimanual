"""Run the bimanual T1 controller in MuJoCo's live interactive viewer."""

from __future__ import annotations

import argparse
import time

import mujoco.viewer

from mise_en_place import config as C
from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.control.scripted_t2 import ScriptedT2Handoff
from mise_en_place.sim.env import DinnerTableEnv


class ViewerClosed(RuntimeError):
    """Raised when the user closes the live viewer during an episode."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t1", "t2"), default="t1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Auto-close this many seconds after completion (default: wait for user)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use the slower sequential fallback instead of true parallel T1",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")

    env = DinnerTableEnv()
    env.reset(seed=args.seed)
    expert = ScriptedT1Expert(env) if args.task == "t1" else ScriptedT2Handoff(env)
    original_step = env.step
    wall_step = (1.0 / C.CONTROL_HZ) / args.speed

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.sync()

        def live_step(action, cameras=()):
            if not viewer.is_running():
                raise ViewerClosed
            started = time.perf_counter()
            obs = original_step(action, cameras)
            viewer.sync()
            remaining = wall_step - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            return obs

        env.step = live_step  # type: ignore[method-assign]
        try:
            if args.task == "t1":
                score = expert.run_parallel_t1(true_parallel=not args.sequential)
            else:
                score = expert.run_t2()
            print(score)
            if args.hold_seconds is None:
                print("Episode complete. Close the viewer window to exit.")
            completed_at = time.perf_counter()
            while viewer.is_running() and (
                args.hold_seconds is None
                or time.perf_counter() - completed_at < args.hold_seconds
            ):
                viewer.sync()
                time.sleep(0.05)
        except ViewerClosed:
            print("Viewer closed; episode stopped.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
