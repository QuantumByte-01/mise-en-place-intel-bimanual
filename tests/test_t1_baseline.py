"""Closed-loop check of the scripted geometric T1 baseline on a few randomized seeds."""

import pytest

from mise_en_place.control.scripted_t1 import ScriptedT1Expert
from mise_en_place.sim.env import DinnerTableEnv


@pytest.fixture(scope="module")
def env():
    e = DinnerTableEnv()
    yield e
    e.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_t1_places_both_utensils(env, seed):
    env.reset(seed=seed)
    score = ScriptedT1Expert(env).run_parallel_t1()
    assert score["success"], score
