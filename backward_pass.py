import jax.numpy as jnp
import jax.lax
from jax import jacrev, grad, hessian
import jax.scipy as jcp


def bwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    goal_state: jnp.ndarray,
    nominal_slacks: jnp.ndarray,
    nominal_duals: jnp.ndarray,
    barrier_param: float,
    reg_param: float,
    dynamics,
    constraints,
    final_cost,
    stage_Lagrangian,
):
    def bwd_step(carry, inp):
        # unpack carry
        Vx, Vxx, pos_def = carry

        # state, controls, dual
        state, control, slack, dual = inp

        # compute derivatives at nominal_values
        lx = grad(stage_Lagrangian, 0)(state, control, goal_state, dual)
        fx = jacrev(dynamics, 0)(state, control)
        lu = grad(stage_Lagrangian, 1)(state, control, goal_state, dual)
        fu = jacrev(dynamics, 1)(state, control)
        lxx = hessian(stage_Lagrangian, 0)(state, control, goal_state, dual)
        luu = hessian(stage_Lagrangian, 1)(state, control, goal_state, dual)
        lxu = jacrev(jacrev(stage_Lagrangian, 0), 1)(state, control, goal_state, dual)
        cx = jacrev(constraints, 0)(state, control)
        cu = jacrev(constraints, 1)(state, control)
        fxx = jacrev(jacrev(dynamics, 0), 0)(state, control)
        fuu = jacrev(jacrev(dynamics, 1), 1)(state, control)
        fxu = jacrev(jacrev(dynamics, 0), 1)(state, control)

        # Q function expansion coeffs
        Qx = lx + fx.T @ Vx
        Qu = lu + fu.T @ Vx
        Qxx = lxx + fx.T @ Vxx @ fx + jnp.tensordot(Vx, fxx, axes=1)
        Quu = luu + fu.T @ Vxx @ fu + jnp.tensordot(Vx, fuu, axes=1)
        Quu += reg_param * jnp.eye(control.shape[0])
        Qxu = lxu + fx.T @ Vxx @ fu + jnp.tensordot(Vx, fxu, axes=1)
        Quu = (Quu + Quu.T) / 2.0
        pos_def = jnp.logical_and(jnp.all(jnp.linalg.eigvals(Quu) > 0), pos_def)

        # perturbed KKT
        Slack_inv = jnp.diag(1 / slack)
        Sigma = Slack_inv @ jnp.diag(dual)
        rp = constraints(state, control) + slack
        rd = dual * slack - barrier_param
        r = dual * rp - rd
        A = Quu + cu.T @ Sigma @ cu

        def convex():
            chol_and_lower = jcp.linalg.cho_factor(A)
            alpha = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ r - Qu)
            beta = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ cx - Qxu.T)
            eta = Sigma @ cu @ alpha + Slack_inv @ r
            theta = Sigma @ (cu @ beta + cx)
            chi = -rp - cu @ alpha
            zeta = -(cu @ beta + cx)
            return alpha, beta, eta, theta, chi, zeta

        def indef():
            return (
                jnp.zeros_like(control),
                jnp.zeros((control.shape[0], state.shape[0])),
                jnp.zeros_like(dual),
                jnp.zeros((dual.shape[0], state.shape[0])),
                jnp.zeros_like(slack),
                jnp.zeros((slack.shape[0], state.shape[0])),
            )

        # return petrubed KKT system solution if the minimization problem is convex
        k_control, K_control, k_dual, K_dual, k_slack, K_slack = jax.lax.cond(
            pos_def, convex, indef
        )

        # update Q function expansion parameters
        Qx = Qx + cx.T @ Slack_inv @ r
        Qu = Qu + cu.T @ Slack_inv @ r
        Qxx = Qxx + cx.T @ Sigma @ cx
        Qxu = Qxu + cx.T @ Sigma @ cu
        Quu = Quu + cu.T @ Sigma @ cu

        # Value function expansion parameters and diff cost
        Vx = Qx + Qxu @ k_control
        Vxx = Qxx + Qxu @ K_control
        dV = k_control.T @ Qu + 0.5 * k_control.T @ Quu @ k_control.T

        error = jnp.hstack((Qu, rp, rd))
        return (Vx, Vxx, pos_def), (
            k_control,
            k_dual,
            k_slack,
            K_control,
            K_dual,
            K_slack,
            dV,
            error,
        )

    xN = nominal_states[-1]
    VxN = grad(final_cost, 0)(xN, goal_state)
    VxxN = hessian(final_cost, 0)(xN, goal_state)

    feasible = jnp.bool_(1.0)

    carry_out, bwd_pass_out = jax.lax.scan(
        bwd_step,
        (VxN, VxxN, feasible),
        (nominal_states[:-1], nominal_controls, nominal_slacks, nominal_duals),
        reverse=True,
    )
    (
        control_ff_gain,
        dual_ff_gain,
        slack_ff_gain,
        control_gain,
        dual_gain,
        slack_gain,
        diff_cost,
        optimality_error,
    ) = bwd_pass_out
    diff_cost = jnp.sum(diff_cost)
    _, _, feasible = carry_out
    return (
        control_ff_gain,
        dual_ff_gain,
        slack_ff_gain,
        control_gain,
        dual_gain,
        slack_gain,
        diff_cost,
        jnp.max(jnp.abs(optimality_error)),
        feasible,
    )
