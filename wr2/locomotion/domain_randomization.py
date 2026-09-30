"""Batched MJX dynamics randomization for WR2."""

from __future__ import annotations

from functools import partial

import jax
from brax import base

from wr2.locomotion.configs import DynamicsRandomization


def make_domain_randomizer(
    config: DynamicsRandomization,
):
    """Create a Brax-compatible relative randomization function.

    These broad ranges are suitable for simulation development. They must be
    replaced by distributions fitted from WR2 measurements before deployment.
    """

    def domain_randomize(
        system: base.System, rng: jax.Array
    ) -> tuple[base.System, base.System]:
        @partial(jax.vmap)
        def randomize_one(key: jax.Array):
            keys = jax.random.split(key, 7)
            friction_scale = jax.random.uniform(
                keys[0],
                minval=config.friction_scale[0],
                maxval=config.friction_scale[1],
            )
            damping_scale = jax.random.uniform(
                keys[1],
                (system.nv - 6,),
                minval=config.damping_scale[0],
                maxval=config.damping_scale[1],
            )
            armature_scale = jax.random.uniform(
                keys[2],
                (system.nv - 6,),
                minval=config.armature_scale[0],
                maxval=config.armature_scale[1],
            )
            frictionloss_scale = jax.random.uniform(
                keys[3],
                (system.nv - 6,),
                minval=config.frictionloss_scale[0],
                maxval=config.frictionloss_scale[1],
            )
            kp_scale = jax.random.uniform(
                keys[4],
                (system.nu,),
                minval=config.kp_scale[0],
                maxval=config.kp_scale[1],
            )
            kv_scale = jax.random.uniform(
                keys[5],
                (system.nu,),
                minval=config.kv_scale[0],
                maxval=config.kv_scale[1],
            )
            torque_limit_scale = jax.random.uniform(
                keys[6],
                (system.nu, 1),
                minval=config.torque_limit_scale[0],
                maxval=config.torque_limit_scale[1],
            )

            geom_friction = system.geom_friction.at[:, 0].multiply(friction_scale)
            dof_damping = system.dof_damping.at[6:].multiply(damping_scale)
            dof_armature = system.dof_armature.at[6:].multiply(armature_scale)
            dof_frictionloss = system.dof_frictionloss.at[6:].multiply(
                frictionloss_scale
            )
            actuator_gainprm = system.actuator_gainprm.at[:, 0].multiply(kp_scale)
            actuator_biasprm = system.actuator_biasprm.at[:, 1].multiply(kp_scale)
            actuator_biasprm = actuator_biasprm.at[:, 2].multiply(kv_scale)
            actuator_forcerange = system.actuator_forcerange * torque_limit_scale
            return (
                geom_friction,
                dof_damping,
                dof_armature,
                dof_frictionloss,
                actuator_gainprm,
                actuator_biasprm,
                actuator_forcerange,
            )

        (
            geom_friction,
            dof_damping,
            dof_armature,
            dof_frictionloss,
            actuator_gainprm,
            actuator_biasprm,
            actuator_forcerange,
        ) = randomize_one(rng)

        replacements = {
            "geom_friction": geom_friction,
            "dof_damping": dof_damping,
            "dof_armature": dof_armature,
            "dof_frictionloss": dof_frictionloss,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "actuator_forcerange": actuator_forcerange,
        }
        in_axes = jax.tree.map(lambda _: None, system).tree_replace(
            {name: 0 for name in replacements}
        )
        return system.tree_replace(replacements), in_axes

    return domain_randomize
