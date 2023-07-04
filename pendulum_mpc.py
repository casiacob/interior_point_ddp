import jax.numpy as jnp
from utils import discretize_dynamics, wrap_angle
import jax
from jax.config import config
import matplotlib.pyplot as plt
from ipddp import infeasible_ipddp

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([2e0, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state

    c = 0.5 * _wrapped.T @ final_state_cost @ _wrapped
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([2e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state

    c = 0.5 * _wrapped.T @ state_cost @ _wrapped
    c += 0.5 * action.T @ action_cost @ action
    return c


def constraints(state: jnp.ndarray, control: jnp.ndarray):
    control_ub = 5.0
    control_lb = -5.0
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def pendulum(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length**2),
        )
    )


simulation_step = 0.01
downsampling = 5
dynamics = discretize_dynamics(
    ode=pendulum, simulation_step=simulation_step, downsampling=downsampling
)


def plot_traj(states, controls):
    plt.plot(states[:, 0], label="angle")
    plt.plot(states[:, 1], label="angular velocity")
    plt.legend()
    plt.show()
    plt.plot(controls)
    plt.show()


horizon = 40
mean = jnp.array([0.0])
sigma = jnp.array([0.01])
key = jax.random.PRNGKey(3)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.01), -0.01])
xd = jnp.array((jnp.pi, 0.0))
z = 0.01 * jnp.ones((horizon, 2))
s = 0.1 * jnp.ones((horizon, 2))


def mpc_body(carry, inp):
    prev_state, prev_control = carry
    jax.debug.print("initial state = {x}", x=prev_state)
    states, control, _, _ = infeasible_ipddp(
        prev_state, prev_control, xd, s, z, dynamics, constraints, transient_cost, final_cost
    )
    next_state = dynamics(prev_state, control[0])
    return (next_state, control), (next_state, control[0])


_, mpc_out = jax.lax.scan(mpc_body, (x0, u), None, length=100)
mpc_states, mpc_controls = mpc_out
mpc_states = jnp.vstack((x0, mpc_states))
plot_traj(mpc_states, mpc_controls)