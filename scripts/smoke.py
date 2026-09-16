"""Entry: rebuild scene, load env, save demo camera frame."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from mise_en_place import config as C
from mise_en_place.sim.env import DinnerTableEnv
from mise_en_place.sim.scene_builder import build_scene


def main() -> None:
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = build_scene()
    print(f"scene: {path}")
    env = DinnerTableEnv(rebuild=False)
    obs = env.reset(seed=0, randomize=C.SceneRandomization(object_xy=False, plate_xy=False, lighting=False))
    print(f"nq={env.model.nq} nu={env.model.nu} neq={env.model.neq}")
    print(f"tcp L={obs.tcp['left']} R={obs.tcp['right']}")
    print(f"objects={ {k: v.round(3).tolist() for k, v in obs.objects.items()} }")
    # hold home for a few steps
    home = np.tile(np.array(C.HOME_QPOS, dtype=float), 2)
    for _ in range(20):
        env.step(home)
    frame = env.render("demo")
    out = C.ARTIFACTS / "smoke_demo.png"
    imageio.imwrite(out, frame)
    print(f"wrote {out} shape={frame.shape}")
    env.close()


if __name__ == "__main__":
    main()
