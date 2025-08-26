use std::process::exit;

use pyo3::prelude::*;
use tch::{Kind, Tensor};

use crate::network::DqnAgent;

pub fn run_inference(py: Python) -> PyResult<()> {
    let gym = py.import_bound("gymnasium")?;
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("render_mode", "human")?;

    let env = gym.call_method("make", ("CartPole-v1",), Some(&kwargs))?;

    let state_dim = env
        .getattr("observation_space")?
        .getattr("shape")?
        .get_item(0)?
        .extract::<i64>()?;

    let action_dim = env
        .getattr("action_space")?
        .getattr("n")?
        .extract::<i64>()?;

    let mut agent = DqnAgent::new(state_dim, action_dim);

    println!("Loading Trained model from dqn_agent_cartpole_ot...");
    agent.var_store.load("dqn_agent_cartpole.ot").unwrap();

    agent.epsilon = 0.0;

    for episode in 0..15 {
        let (mut state_vec, _info): (Vec<f64>, PyObject) = env.call_method0("reset")?.extract()?;
        let mut total_reward = 0.0;

        loop {
            let state_tensor = Tensor::from_slice(&state_vec)
                .to_kind(Kind::Float)
                .unsqueeze(0)
                .to(agent.device);

            let action = agent.select_action(&state_tensor);

            let (next_state_vec, reward, terminated, truncated, _info): (
                Vec<f64>,
                f64,
                bool,
                bool,
                PyObject,
            ) = env.call_method1("step", (action,))?.extract()?;
            let done = terminated || truncated;
            state_vec = next_state_vec;
            total_reward += reward;

            if done {
                break;
            }
        }
        println!(
            "Inference Episode: {}, Total Reward: {}",
            episode, total_reward
        );
    }

    env.call_method0("close")?;
    exit(1);
}
