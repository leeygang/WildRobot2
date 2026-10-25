"""Direct checkpoint execution through native policy, physics and viewer frames."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks

from wr2.locomotion.configs import load_training_config, training_config_to_dict
from wr2.locomotion.ppo import make_network_factory
from wr2.tools import view_walking
from wr2.tools.diagnose_walking import target_trace


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    run = tmp_path_factory.mktemp("live_policy")
    config = load_training_config(
        Path(__file__).resolve().parents[1] / "wr2/locomotion/configs/ppo_walking_progress.yaml"
    )
    saved = training_config_to_dict(config)
    saved["environment"]["episode_length"] = 20
    (run / "training_config.yaml").write_text(yaml.safe_dump(saved))
    network = make_network_factory(config.network)({"state": 975, "privileged_state": 1485}, 12)
    params = (
        running_statistics.init_state({"state": jp.zeros(975), "privileged_state": jp.zeros(1485)}),
        network.policy_network.init(jax.random.PRNGKey(13)),
        network.value_network.init(jax.random.PRNGKey(14)),
    )
    brax_model.save_params(run / "best_params", params)
    loaded = view_walking.LivePolicy(run)
    loaded.expected_policy = jax.jit(networks.make_inference_fn(network)(params[:2], deterministic=True))
    return loaded


def test_loaded_policy_uses_native_step_delay_and_frame_sampling(live):
    live.reset()
    before = live.state
    action = live.expected_policy(before.obs, before.info["rng"])[0]
    assert np.max(np.abs(action)) > 1e-4
    initial_target = target_trace(live.env, before, live.env.home_action)[1]
    expected = jax.jit(live.env.step)(before, action)
    live.advance()
    np.testing.assert_allclose(live.state.pipeline_state.q, expected.pipeline_state.q, atol=1e-6)
    np.testing.assert_allclose(live.state.info["previous_action"], action, atol=1e-7)
    # First control tick uses reset's previous action; the next uses the actor.
    active = live.env._active_indices
    np.testing.assert_allclose(live.trace["delivered_rad"][0], initial_target[active], atol=1e-7)
    before = live.state
    action = live.expected_policy(before.obs, before.info["rng"])[0]
    expected_target = target_trace(live.env, before, action)[1]
    live.advance()
    np.testing.assert_allclose(live.trace["delivered_rad"][0], expected_target[active], atol=1e-7)
    np.testing.assert_array_equal(live.trace["qpos"][0], live.state.pipeline_state.q)
    np.testing.assert_array_equal(live.trace["foot_geom_center_m"][0], live.state.pipeline_state.geom_xpos[live.env._foot_geom_ids])
    assert live.time_s == pytest.approx(.04)
    assert not live.env.add_observation_noise
    assert not live.env.config.randomization.enabled
    np.testing.assert_allclose(live.trace["command_m_s"][0], [.1 * live.env.config.normalization_scale, 0., 0.])


def test_live_timeout_and_restart_keep_command_and_clear_support_age(live):
    live.reset()
    initial = np.asarray(live.state.pipeline_state.q)
    for _ in range(20):
        live.advance()
    assert live.time_s == 0
    np.testing.assert_array_equal(live.trace["support_age_s"], [[0., 0.]])
    np.testing.assert_array_equal(live.state.info["command"], live.command)
    live.reset()
    np.testing.assert_array_equal(live.state.pipeline_state.q, initial)


def test_live_viewer_updates_frames_without_trace_file(live, monkeypatch):
    live.reset()
    args = view_walking.parse_args(["--run-dir", str(live.path.parent)])
    monkeypatch.setattr(view_walking, "LivePolicy", lambda *args: live)
    monkeypatch.setattr(view_walking, "glfw", SimpleNamespace(init=lambda: True, get_monitors=lambda: [object()]))
    frame_count = [0]
    texts = []
    viewer = SimpleNamespace(
        lock=nullcontext,
        opt=SimpleNamespace(geomgroup=np.ones(6), sitegroup=np.ones(6)),
        cam=SimpleNamespace(lookat=np.zeros(3), distance=0., azimuth=0., elevation=0.),
        user_scn=mujoco.MjvScene(live.model, maxgeom=100),
        is_running=lambda: frame_count[0] < 3,
        set_texts=lambda value: texts.append(value),
        sync=lambda: frame_count.__setitem__(0, frame_count[0] + 1),
    )
    monkeypatch.setattr(view_walking.mujoco.viewer, "launch_passive", lambda *args, **kwargs: nullcontext(viewer))
    ticks = iter([0., .1, .2, .3])
    monkeypatch.setattr(view_walking, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None))
    assert view_walking.view_trace(args) == 0
    assert live.time_s == pytest.approx(.06)
    assert all("Live policy" in text[2] and "best_params" in text[3] for text in texts)
    assert viewer.user_scn.ngeom == 24


def test_headless_viewer_fails_before_loading_policy(monkeypatch):
    monkeypatch.setattr(view_walking, "glfw", SimpleNamespace(init=lambda: True, get_monitors=lambda: []))
    args = view_walking.parse_args(["--run-dir", "missing-run"])
    with pytest.raises(RuntimeError, match="desktop display"):
        view_walking.view_trace(args)


@pytest.mark.parametrize("arguments", [
    [],
    ["--trace", "trace.npz", "--run-dir", "run"],
    ["--trace", "trace.npz", "--checkpoint", "params"],
    ["--run-dir", "run", "--robot", "toddlerbot"],
    ["--run-dir", "run", "--command", "nan", "0", "0"],
])
def test_invalid_live_sources_and_commands_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        view_walking.parse_args(arguments)
    assert error.value.code == 2
