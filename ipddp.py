import jax.numpy as jnp
import jax
from backward_pass import bwd_pass
from forward_pass import fwd_pass


def infeasible_ipddp(
    initial_state0: jnp.ndarray,
    initial_controls: jnp.ndarray,
    goal_state: jnp.ndarray,
    initial_slacks: jnp.ndarray,
    initial_duals: jnp.ndarray,
    dynamics,
    constraints,
    stage_cost,
    final_cost,
):
    def barrier_stage_cost(
        state: jnp.ndarray,
        control: jnp.ndarray,
        goal: jnp.ndarray,
        slack: jnp.ndarray,
        barrier_param: float,
    ):
        return stage_cost(state, control, goal) - barrier_param * jnp.sum(
            jnp.log(slack)
        )

    def barrier_total_cost(
        states: jnp.ndarray,
        controls: jnp.ndarray,
        goal: jnp.ndarray,
        slack,
        barrier_param: float,
    ):
        J = jax.vmap(barrier_stage_cost, in_axes=(0, 0, None, 0, None))(
            states[:-1], controls, goal, slack, barrier_param
        )
        return jnp.sum(J) + final_cost(states[-1], goal)

    def stage_Lagrangian(
        state: jnp.ndarray, control: jnp.ndarray, goal: jnp.ndarray, dual: jnp.ndarray
    ):
        return stage_cost(state, control, goal) + dual.T @ constraints(state, control)

    def ipddp_iteration(val):
        # unpack iteration values
        (
            states,
            controls,
            slacks,
            duals,
            reg_param,
            reg_param_mult_fact,
            barrier_param,
            _,
            loop_counter,
        ) = val

        # compute barrier problem cost at current values
        cost = barrier_total_cost(states, controls, goal_state, slacks, barrier_param)
        jax.debug.print("Cost:            {x}", x=cost)

        # run backward pass
        (
            control_ff_gain,
            dual_ff_gain,
            slack_ff_gain,
            control_gain,
            dual_gain,
            slack_gain,
            diff_cost,
            opt_err_bp,
            feasible_bp,
        ) = bwd_pass(
            states,
            controls,
            goal_state,
            slacks,
            duals,
            barrier_param,
            reg_param,
            dynamics,
            constraints,
            final_cost,
            stage_Lagrangian,
        )

        # run forward pass
        new_states, new_controls, new_slacks, new_duals, feasible_fp = fwd_pass(
            states,
            controls,
            slacks,
            duals,
            control_ff_gain,
            slack_ff_gain,
            dual_ff_gain,
            control_gain,
            slack_gain,
            dual_gain,
            dynamics,
        )

        # compute cost at new values
        new_cost = barrier_total_cost(
            new_states, new_controls, goal_state, new_slacks, barrier_param
        )
        jax.debug.print("New Cost:        {x}", x=new_cost)

        # compute gain ratio: actual reduction/predicted reduction
        val_change = cost - new_cost
        gain_ratio = val_change / (-diff_cost)

        def accept_step():
            return (
                reg_param * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                2.0,
                new_states,
                new_controls,
                new_slacks,
                new_duals,
            )

        def reject_step():
            return (
                reg_param * reg_param_mult_fact,
                reg_param_mult_fact * 2.0,
                states,
                controls,
                slacks,
                duals,
            )

        # accept or reject step and update the regularization parameter
        reg_param, reg_param_mult_fact, states, controls, slacks, duals = jax.lax.cond(
            jnp.logical_and(gain_ratio > 0.0, feasible_bp),
            accept_step,
            reject_step,
        )

        # check the optimality error and update the barrier parameter
        barrier_param = jnp.where(
            jnp.logical_and(opt_err_bp < 0.2 * barrier_param, feasible_bp),
            jnp.min(jnp.array([0.2 * barrier_param, barrier_param**1.2])),
            barrier_param,
        )
        loop_counter += 1
        jax.debug.print("Feasible bwd     {x}", x=feasible_bp)
        jax.debug.print("Feasible fwd     {x}", x=feasible_fp)
        jax.debug.print("slacks > 0       {x}", x=jnp.all(slacks > 0))
        jax.debug.print("duals > 0        {x}", x=jnp.all(duals > 0))
        jax.debug.print("bp:              {x}", x=barrier_param)
        jax.debug.print("opt err:         {x}", x=opt_err_bp)
        jax.debug.print("reg param:       {x}", x=reg_param)
        jax.debug.print("------------------------")
        return (
            states,
            controls,
            slacks,
            duals,
            reg_param,
            reg_param_mult_fact,
            barrier_param,
            opt_err_bp,
            loop_counter,
        )

    def ipddp_convergence(val):
        _, _, _, _, reg_param, _, bp, opt_err, loop_counter = val
        exit_condition = jnp.logical_or(
            jnp.maximum(opt_err, bp) < 1e-6, reg_param == jnp.inf
        )
        # exit_condition = bp / initial_barrier_param < 1e-6
        return jnp.logical_not(exit_condition)

    # initialize regularization parameter and multiplication factor
    initial_reg_param = 1.
    initial_reg_param_mult_fact = 2.0

    # run the dynamics on the initial controls and initial state for initial states
    def body_scan(prev_state, ctrl):
        return dynamics(prev_state, ctrl), dynamics(prev_state, ctrl)

    _, initial_states = jax.lax.scan(body_scan, initial_state0, initial_controls)
    initial_states = jnp.vstack((initial_state0, initial_states))

    # initialize barrier parameter
    horizon = initial_controls.shape[0]
    n_c = constraints(initial_states[0], initial_controls[0]).shape[0]
    initial_barrier_param = barrier_total_cost(
        initial_states, initial_controls, goal_state, initial_slacks, 0.0
    ) / (horizon * n_c)

    # run infeasible ipddp
    (
        optimal_states,
        optimal_controls,
        optimal_slacks,
        optimal_duals,
        reg_param_out,
        _,
        _,
        _,
        iterations,
    ) = jax.lax.while_loop(
        ipddp_convergence,
        ipddp_iteration,
        (
            initial_states,
            initial_controls,
            initial_slacks,
            initial_duals,
            initial_reg_param,
            initial_reg_param_mult_fact,
            initial_barrier_param,
            jnp.inf,
            0,
        ),
    )
    jax.debug.print("ipddp converged in {x}", x=iterations)
    return (
        optimal_states,
        optimal_controls,
        optimal_slacks,
        optimal_duals,
    )
