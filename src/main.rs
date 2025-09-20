//! Main entry point for the Rust DQN example.
//! This file orchestrates the environment interaction, agent training/evaluation,
//! and TensorBoard logging.

use rust_dqn_example::{
    gym_wrapper::GymnasiumWrapper,
    qnetwork::{DQNAgent, Transition},
};
use std::error::Error;
use tensorboard_rs::summary_writer::SummaryWriter;

use rust_dqn_example::qnetwork::AgentConfig;

/// Main function to run the DQN agent.
/// It initializes the environment, agent, and TensorBoard writer.
/// It then proceeds with either training or evaluation based on the `to_train` flag.
fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the CartPole environment with RGB array rendering for potential visualization.
    let mut cartpole = GymnasiumWrapper::new("rgb_array".to_string())?;

    // Define the configuration for the DQN agent.
    // These hyperparameters control the learning process and replay buffer.
    let agent_config = AgentConfig {
        learning_rate: 1e-4,
        gamma: 0.99,
        epsilon_decay: 0.999,
        epsilon_min: 0.01,
        replay_buffer_capacity: 50000,
    };

    // Create a new DQN agent with the specified configuration.
    let mut dqn_agent = DQNAgent::new(agent_config);

    // Initialize TensorBoard SummaryWriter for logging training progress.
    // Logs will be saved in the 'logdir' directory.
    let mut writer = SummaryWriter::new("./logdir");

    // Define training and evaluation parameters.
    let batch_size = 128;
    let episodes = 100;
    let target_update_frequency = 1000;
    // Set `to_train` to `true` for training, `false` for evaluation/test run.
    let to_train = false;
    // Set epsilon to 0.0 for evaluation to ensure deterministic, greedy actions.
    dqn_agent.epsilon = 0.00;

    // Construct the path for saving/loading the agent's model.
    let model = format!("{}/alpha.ot", env!("CARGO_MANIFEST_DIR"));

    // Training loop: Executes if `to_train` is true.
    if to_train {
        // Load a pre-trained agent if the model file exists, for continued training.
        if std::path::Path::new(&model).exists() {
            println!(
                "Loading saved agent for continued training from {}...",
                model
            );
            dqn_agent.load(&model)?;
        }

        // Iterate through the specified number of training episodes.
        for episode in 0..episodes {
            let mut state = cartpole.reset()?;
            let mut total_reward = 0.0;

            // Each episode runs for a maximum of 500 steps.
            let steps = 500;

            // Step through the environment until done or max steps reached.
            for _step in 0..steps {
                // Agent selects an action based on the current state (epsilon-greedy).
                let action = dqn_agent.action(&state);
                // Execute the action in the environment and get the next state, reward, and done flag.
                let (next_state, reward, done) = cartpole.step(action)?;

                // Store the transition in the replay buffer.
                dqn_agent.replay_buffer.push(Transition {
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                });
                // Update the current state and accumulate the total reward for the episode.
                state = next_state;
                total_reward += reward;

                // Train the DQN agent using a batch of experiences from the replay buffer.
                dqn_agent.train(batch_size);

                // Update the target network periodically to stabilize training.
                if dqn_agent.step_counter % target_update_frequency == 0 {
                    dqn_agent.update_target_network();
                }

                // Break the episode loop if the environment indicates it's done.
                if done {
                    break;
                }
            }

            // Print training progress every 25 episodes.
            if episode % 25 == 0 {
                println!(
                    "Episode: {}, Total Reward: {:.2}, Epsilon: {:.3}",
                    episode, total_reward, dqn_agent.epsilon
                )
            }
            // Log the total reward for the episode to TensorBoard.
            writer.add_scalar("rewards/total_reward", total_reward, episode as usize);
        }
        // Ensure all TensorBoard events are written to disk after training.
        writer.flush();

        // Save the trained agent's model.
        println!("訓練終了！エージェントを保存します。。。");
        dqn_agent.save(&model)?;
        println!("保存しました！");
    } else {
        // Evaluation loop: Executes if `to_train` is false.
        // Load a saved agent for evaluation if the model file exists.
        if std::path::Path::new(&model).exists() {
            println!("Loading saved agent from {}...", model);
            dqn_agent.load(&model)?;
        }
        // Set epsilon to 0.0 for evaluation to ensure deterministic, greedy actions.
        dqn_agent.epsilon = 0.00;
        let mut state = cartpole.reset()?;
        let mut total_reward = 0.0;

        // Set render mode to "human" for visual observation during evaluation.
        cartpole.set_render_view("human".to_string())?;

        // Run 5 games for evaluation.
        for game in 0..5 {
            cartpole.reset()?;
            let mut reward_game = 0.0;
            // Each game runs for a maximum of 500 steps.
            for _step in 0..500 {
                // Render the environment for visual feedback.
                cartpole.render()?;
                // Agent selects a greedy action based on the current state.
                let action = dqn_agent.action(&state);
                // Execute the action and get the next state, reward, and done flag.
                let (next_state, reward, done) = cartpole.step(action)?;

                // Update the current state and accumulate rewards.
                state = next_state;
                total_reward += reward;
                reward_game += reward;
                // Break if the game is done.
                if done {
                    break;
                }
            }
            // Print the reward for the current game.
            println!("Reward {} for game {}", reward_game, game);
        }

        // Print the total reward accumulated over all evaluation games.
        println!("Episode finished with a total reward of: {}", total_reward);
        // Close the environment rendering window.
        cartpole.close()?;
    }

    Ok(())
}
