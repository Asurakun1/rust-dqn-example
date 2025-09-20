use rust_dqn_example::{
    gym_wrapper::GymnasiumWrapper,
    qnetwork::{DQNAgent, Transition},
};
use std::error::Error;
use tensorboard_rs::summary_writer::SummaryWriter;

use rust_dqn_example::qnetwork::AgentConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let mut cartpole = GymnasiumWrapper::new("rgb_array".to_string())?;

    let agent_config = AgentConfig {
        learning_rate: 1e-4,
        gamma: 0.99,
        epsilon_decay: 0.999,
        epsilon_min: 0.01,
        replay_buffer_capacity: 50000,
    };

    let mut dqn_agent = DQNAgent::new(agent_config);

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    let batch_size = 128;
    let episodes = 100;
    let target_update_frequency = 1000;
    let to_train = false;
    dqn_agent.epsilon = 0.00;

    let model = format!("{}/alpha.ot", env!("CARGO_MANIFEST_DIR"));

    if to_train {
        if std::path::Path::new(&model).exists() {
            println!(
                "Loading saved agent for continued training from {}...",
                model
            );
            dqn_agent.load(&model)?;
        }
        for episode in 0..episodes {
            let mut state = cartpole.reset()?;
            let mut total_reward = 0.0;

            let steps = 500;

            for _step in 0..steps {
                let action = dqn_agent.action(&state);
                let (next_state, reward, done) = cartpole.step(action)?;

                dqn_agent.replay_buffer.push(Transition {
                    state: state.clone(),
                    action,
                    reward,
                    next_state,
                    done,
                });
                state = next_state;
                total_reward += reward;

                dqn_agent.train(batch_size);

                if dqn_agent.step_counter % target_update_frequency == 0 {
                    dqn_agent.update_target_network();
                }

                if done {
                    break;
                }
            }

            if episode % 25 == 0 {
                println!(
                    "Episode: {}, Total Reward: {:.2}, Epsilon: {:.3}",
                    episode, total_reward, dqn_agent.epsilon
                )
            }
            writer.add_scalar("rewards/total_reward", total_reward, episode as usize);
        }
        writer.flush();

        println!("訓練終了！エージェントを保存します。。。");
        dqn_agent.save(&model)?;
        println!("保存しました！");
    } else {
        if std::path::Path::new(&model).exists() {
            println!("Loading saved agent from {}...", model);
            dqn_agent.load(&model)?;
        }
        dqn_agent.epsilon = 0.00;
        let mut state = cartpole.reset()?;
        let mut total_reward = 0.0;

        cartpole.set_render_view("human".to_string())?;

        for game in 0..5 {
            cartpole.reset()?;
            let mut reward_game = 0.0;
            for _step in 0..500 {
                cartpole.render()?;
                let action = dqn_agent.action(&state);
                let (next_state, reward, done) = cartpole.step(action)?;

                state = next_state;
                total_reward += reward;
                reward_game += reward;
                if done {
                    break;
                }
            }
            println!("Reward {} for game {}", reward_game, game);
        }

        println!("Episode finished with a total reward of: {}", total_reward);
        cartpole.close()?;
    }

    Ok(())
}
