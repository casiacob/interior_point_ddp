import jax.numpy as jnp
from utils import wrap_angle
import jax
from utils import euler
from jax import config
from ipddp import infeasible_ipddp
import time

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

def constraints(state: jnp.ndarray, control: jnp.ndarray):
    control_ub = 50.0
    control_lb = -50.0
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))

Ts = [0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001]
N = [20, 40, 80, 100, 200, 400, 800, 1000]
ipddp_means = []
xd = jnp.array([0.0, jnp.pi, 0.0, 0.0])

for sampling_period, horizon in zip(Ts, N):
    par_time_array = []
    seq_time_array = []
    downsampling = 1
    dynamics = euler(cartpole, sampling_period)

    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(1)
    u = 0.1 * jax.random.normal(key, shape=(horizon, 1))

    z = 0.01 * jnp.ones((horizon, 2))
    s = 0.1 * jnp.ones((horizon, 2))


    annon_ipddp = lambda state0, control, state_d, slack, lag: infeasible_ipddp(
        state0, control, state_d, slack, lag, dynamics, constraints, transient_cost, final_cost
    )

    _jitted_ipddp = jax.jit(annon_ipddp)
    _jitted_ipddp(x0, u, xd, s, z)

    for i in range(10):
        start = time.time()
        states, control, _, _ = _jitted_ipddp(x0, u, xd, s, z)
        jax.block_until_ready(control)
        end = time.time()
        ipddp_time = end - start


        par_time_array.append(ipddp_time)

    ipddp_means.append(jnp.mean(jnp.array(par_time_array)))
par_time_means_arr = jnp.array(ipddp_means)


print(par_time_means_arr)