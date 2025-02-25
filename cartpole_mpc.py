import jax.numpy as jnp
from utils import wrap_angle, discretize_dynamics
import jax
import matplotlib.pyplot as plt
from jax import config
from ipddp import infeasible_ipddp

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e1, 1e2, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([1e1, 1e2, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def cartpole(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


simulation_step = 0.05
downsampling = 1
dynamics = discretize_dynamics(
    ode=cartpole, simulation_step=simulation_step, downsampling=downsampling
)


def constraints(state: jnp.ndarray, control: jnp.ndarray):
    control_ub = 50.0
    control_lb = -50.0
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def plot_traj(states, controls):
    plt.plot(states[:, 0], label="cart position")
    plt.plot(states[:, 1], label="angle position")
    plt.legend()
    plt.show()
    plt.plot(states[:, 2], label="cart velocity")
    plt.plot(states[:, 3], label="angle velocity")
    plt.legend()
    plt.show()
    plt.plot(controls)
    plt.show()


horizon = 20
mean = jnp.array([0.0])
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 1))
x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
xd = jnp.array([0.0, jnp.pi, 0.0, 0.0])
z = 0.01 * jnp.ones((horizon, 2))
s = 0.1 * jnp.ones((horizon, 2))


def mpc_body(carry, inp):
    prev_state, prev_control = carry
    jax.debug.print(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    jax.debug.print("initial state = {x}", x=prev_state)
    states, control, _, _ = infeasible_ipddp(
        prev_state,
        prev_control,
        xd,
        s,
        z,
        dynamics,
        constraints,
        transient_cost,
        final_cost,
    )
    next_state = dynamics(prev_state, control[0])
    # jax.debug.callback(print_traj, states, control)
    # jax.debug.breakpoint()
    return (next_state, control), (next_state, control[0])


_, mpc_out = jax.lax.scan(mpc_body, (x0, u), None, length=200)
mpc_states, mpc_controls = mpc_out
mpc_states = jnp.vstack((x0, mpc_states))
plot_traj(mpc_states, mpc_controls)
