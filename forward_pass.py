import jax.numpy as jnp
import jax


def fwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    nominal_slacks: jnp.ndarray,
    nominal_duals: jnp.ndarray,
    control_ff_gain: jnp.ndarray,
    slack_ff_gain,
    dual_ff_gain,
    control_gain: jnp.ndarray,
    slack_gain: jnp.ndarray,
    dual_gain: jnp.ndarray,
    dynamics,
):
    def fwd_step(prev_state, inp):
        (
            state,
            control,
            slack,
            dual,
            k_control,
            k_slack,
            k_dual,
            K_control,
            K_slack,
            K_dual,
        ) = inp

        control = control + k_control + K_control @ (prev_state - state)

        dual_direction = k_dual + K_dual @ (prev_state - state)
        max_dual_step = jnp.where(
            -0.995 * dual / dual_direction > 0.0, -0.995 * dual / dual_direction, 0
        )
        max_dual_step = jnp.min(max_dual_step, initial=1.0, where=max_dual_step > 0)
        dual = dual + max_dual_step * dual_direction

        slack_direction = k_slack + K_slack @ (prev_state - state)
        max_slack_step = jnp.where(
            -0.995 * slack / slack_direction > 0.0, -0.995 * slack / slack_direction, 0
        )
        max_slack_step = jnp.min(max_slack_step, initial=1.0, where=max_slack_step > 0)
        slack = slack + max_slack_step * slack_direction

        next_state = dynamics(prev_state, control)

        feasible = jnp.logical_and(jnp.all(dual > 0), jnp.all(slack > 0))

        return next_state, (next_state, control, slack, dual, feasible)

    _, fwd_pass_out = jax.lax.scan(
        fwd_step,
        nominal_states[0],
        (
            nominal_states[:-1],
            nominal_controls,
            nominal_slacks,
            nominal_duals,
            control_ff_gain,
            slack_ff_gain,
            dual_ff_gain,
            control_gain,
            slack_gain,
            dual_gain,
        ),
    )
    (
        new_states,
        new_controls,
        new_slacks,
        new_duals,
        feasible_fp,
    ) = fwd_pass_out
    new_states = jnp.vstack((nominal_states[0], new_states))
    return (
        new_states,
        new_controls,
        new_slacks,
        new_duals,
        jnp.all(feasible_fp),
    )
