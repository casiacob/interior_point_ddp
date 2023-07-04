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
    final_state_cost = jnp.diag(jnp.array([1e1, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state

    c = 0.5 * _wrapped.T @ final_state_cost @ _wrapped
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([1e1, 1e-1]))
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

horizon = 100
mean = jnp.array([0.0])
sigma = jnp.array([0.01])
key = jax.random.PRNGKey(1235)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([wrap_angle(0.01), -0.01])
xd = jnp.array((jnp.pi, 0.0))
z = 0.01 * jnp.ones((horizon, 2))
s = 0.1 * jnp.ones((horizon, 2))


states, control, _, _ = infeasible_ipddp(
    x0, u, xd, s, z, dynamics, constraints, transient_cost, final_cost
)
plt.plot(states[:, 0])
plt.plot(states[:, 1])
plt.show()
plt.plot(control)
plt.show()