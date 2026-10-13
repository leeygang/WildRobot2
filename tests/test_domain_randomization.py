import copy
import unittest
from dataclasses import replace

import jax
import mujoco
import numpy as np
from brax.io import mjcf

from wr2.locomotion.configs import load_training_config
from wr2.locomotion.domain_randomization import make_domain_randomizer


class DomainRandomizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <default><joint armature="0.02" damping="0.1" frictionloss="0.1"/></default>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body pos="0 0 0.5">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
              <body pos="0 0 -0.15">
                <joint name="hip" axis="0 1 0"/>
                <geom type="capsule" size="0.04 0.1" mass="0.4"/>
                <body pos="0 0 -0.2">
                  <joint name="knee" axis="0 1 0"/>
                  <geom type="capsule" size="0.03 0.1" mass="0.2"/>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator><motor joint="hip"/><motor joint="knee"/></actuator>
        </mujoco>
        """)
        cls.system = mjcf.load_model(cls.model)
        cls.config = load_training_config().environment.randomization

    def test_randomized_constants_match_native_mujoco_after_mass_and_armature(self):
        fields = (
            "body_invweight0", "dof_invweight0", "body_subtreemass",
            "dof_M0", "actuator_acc0", "tendon_invweight0",
        )
        original_mass = self.model.body_mass.copy()
        original_armature = self.model.dof_armature.copy()
        for mass_range in ((0.8, 0.8), (1.2, 1.2), self.config.body_mass_scale):
            with self.subTest(mass_range=mass_range):
                config = replace(self.config, body_mass_scale=mass_range)
                randomized, axes = make_domain_randomizer(config)(
                    self.system, jax.random.split(jax.random.PRNGKey(51), 3)
                )
                oracle = copy.copy(self.model)
                data = mujoco.MjData(oracle)
                for index in range(3):
                    for field in ("body_mass", "body_inertia", "dof_armature"):
                        getattr(oracle, field)[:] = np.asarray(getattr(randomized, field)[index])
                    mujoco.mj_setConst(oracle, data)
                    for field in fields:
                        self.assertEqual(getattr(axes, field), 0)
                        np.testing.assert_allclose(
                            np.asarray(getattr(randomized, field)[index]),
                            getattr(oracle, field), rtol=2e-6, atol=1e-7,
                            err_msg=f"sample {index}: {field}",
                        )
                np.testing.assert_array_equal(self.model.body_mass, original_mass)
                np.testing.assert_array_equal(self.model.dof_armature, original_armature)


if __name__ == "__main__":
    unittest.main()
