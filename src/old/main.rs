use std::fs::File;
use std::io::{self, Write};

use pyo3::types::PyDict;
use rust_dqn_example::run;
use rust_dqn_example::{network::DqnAgent, qnet::ReplayBuffer};
use tch::{Kind, Tensor}; // Import the Optimizer struct specifically

use pyo3::prelude::*;

fn save_rewards_to_csv(rewards: &[f64], filname: &str) -> io::Result<()> {
    let mut file = File::create(filname)?;
    writeln!(&mut file, "episode,reward")?;
    for (i, reward) in rewards.iter().enumerate() {
        writeln!(&mut file, "{},{}", i, reward)?;
    }
    Ok(())
}

fn main() -> PyResult<()> {
    // Agent and Replay Buffer setup is the same
    let buffer_capacity = 10000;
    let batch_size = 64;
    let num_episodes = 10000;

    let mut all_rewards: Vec<f64> = Vec::new();

    // --- PyO3 setup and the main training loop ---
    Python::with_gil(|py| {
        run::run_inference(py)?; //SIMULATION

        // Get state and action dimensions from the environment
        let gym = py.import_bound("gymnasium")?;

        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("render_mode", "rgb_array")?;

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

        // Initialize our Rust agent and buffer
        let mut agent = DqnAgent::new(state_dim, action_dim);
        let mut replay_buffer = ReplayBuffer::new(buffer_capacity);

        // println!("Loading Trained model from dqn_agent_cartpole_ot...");
        // agent.var_store.load("dqn_agent_cartpole.ot").unwrap(); //Load the previous agent

        println!("Starting training on device: {:?}", agent.device);

        for episode in 0..num_episodes {
            let (mut state_vec, _info): (Vec<f64>, PyObject) =
                env.call_method0("reset")?.extract()?;
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

                let next_state_tensor = Tensor::from_slice(&next_state_vec)
                    .to_kind(Kind::Float)
                    .unsqueeze(0)
                    .to(agent.device);
                replay_buffer.push((state_tensor.copy(), action, reward, next_state_tensor, done));

                state_vec = next_state_vec;
                total_reward += reward;

                if replay_buffer.len() > batch_size {
                    let experiences = replay_buffer.sample(batch_size).unwrap();
                    agent.learn(experiences);
                }

                let _ = env.call_method0("render")?;

                if done {
                    break;
                }
            }

            all_rewards.push(total_reward);

            if episode % 10 == 0 {
                println!(
                    "Episode: {}, Total Reward: {}, Epsilon: {:.4}",
                    episode, total_reward, agent.epsilon
                );
            }
        }

        println!("Training complete!");
        env.call_method0("close")?;

        agent.var_store.save("dqn_agent_cartpole.ot").unwrap();
        // The closure needs to return a PyResult

        println!("Saving rewards to rewards.csv...");
        save_rewards_to_csv(&all_rewards, "rewards.csv").expect("Failed to save csv file");

        Ok(())
    }) // The result of the closure is returned here
}
