import jax.numpy as jnp
import jax
from utils import discretize_dynamics
from jax.config import config
from ipddp import infeasible_ipddp
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


def stage_cost(state: jnp.ndarray, control: jnp.ndarray, ref: jnp.ndarray) -> float:
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 1.0, 0.0]))
    control_penalty = jnp.diag(jnp.array([1.0, 1.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    c = c + control.T @ control_penalty @ control
    return c * 0.5


def final_cost(state: jnp.ndarray, ref: jnp.ndarray) -> float:
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 1.0, 0.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    return c * 0.5


def constraints(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    x_pos, y_pos, _, _ = state
    acc, steering = control

    # ellipse obstacle parameters
    ea = 5.0
    eb = 2.5
    xc = 15.0
    yc = -1.0

    # ellipse constraint
    S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    dxy = jnp.array([x_pos - xc, y_pos - yc])
    c0 = 1 - dxy.T @ S @ dxy

    # S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    # dxy = jnp.array([x_pos - 35, y_pos - 1])
    # c5 = 1 - dxy.T @ S @ dxy

    # control bounds
    a_ub = 1.5
    a_lb = -3
    steering_ub = 0.6
    steering_lb = -0.6

    c1 = acc - a_ub
    c2 = a_lb - acc
    c3 = steering - steering_ub
    c4 = steering_lb - steering
    # return jnp.hstack((c0, c1, c2, c3, c4, c5))
    return jnp.hstack((c0, c1, c2, c3, c4))


def car(state: jnp.ndarray, control: jnp.ndarray):
    lf = 1.06
    lr = 1.85
    x, y, v, phi = state
    acceleration, steering = control
    beta = jnp.arctan(jnp.tan(steering * (lr/(lf+lr))))
    return jnp.hstack((
        v * jnp.cos(phi + beta),
        v * jnp.sin(phi + beta),
        acceleration,
        v/lr * jnp.sin(beta)
    ))


simulation_step = 0.1
downsampling = 1
dynamics = discretize_dynamics(
    ode=car, simulation_step=simulation_step, downsampling=downsampling
)

N = 60
mean = jnp.array([0.0, 0.0])
sigma = jnp.array([0.1, 0.0001])
key = jax.random.PRNGKey(2)
x0 = jnp.array([0.0, 0.0, 5.0, 0.0])
xd = jnp.array([0.0, 0.0, 8.0, 0.0])
u = mean + sigma * jax.random.normal(key, shape=(N, 2))
z = 0.1 * jnp.ones((N, 5))
s = 0.1 * jnp.ones((N, 5))


def print_traj(states, controls):
    cx1 = 15.0
    cy1 = -1.0
    # cx2 = 35.0
    # cy2 = 1.0
    a = 5.0  # radius on the x-axis
    b = 2.5  # radius on the y-axis
    t = jnp.linspace(0, 2 * jnp.pi, 150)
    plt.plot(cx1 + a * jnp.cos(t), cy1 + b * jnp.sin(t), color="red")
    # plt.plot(cx2 + a * jnp.cos(t), cy2 + b * jnp.sin(t), color="red")
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel("x position")
    plt.ylabel("y position")

    plt.ylim([-10, 10])
    plt.show()
    plt.plot(states[:, 2])
    plt.ylabel("velocity")
    plt.show()
    plt.plot(states[:, 3])
    plt.ylabel("yaw")
    plt.show()
    plt.plot(controls[:, 0])
    plt.ylabel("acceleration")
    plt.show()
    plt.plot(controls[:, 1])
    plt.ylabel("steering")
    plt.show()


states, control, _, _ = infeasible_ipddp(
    x0, u, xd, s, z, dynamics, constraints, stage_cost, final_cost
)

print_traj(states, control)

