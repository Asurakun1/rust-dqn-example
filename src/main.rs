use std::error::Error;

use pyo3::{prelude::*, types::PyDict};
use rust_dqn_example::{
    environment::{Action, State},
    qnetwork::{DQNAgent, Transition},
};

struct GymnasiumWrapper {
    env: PyObject,
    render_view: String,
    gymnasium: PyObject,
    gym_env: String,
}

impl GymnasiumWrapper {
    pub fn new(render_view: String) -> PyResult<Self> {
        let gym_env = "CartPole-v1".to_string();
        Python::with_gil(|py| {
            let gymnasium = py.import_bound("gymnasium")?;

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("render_mode", render_view.as_str())?;
            let env = gymnasium
                .call_method("make", (&gym_env,), Some(&kwargs))
                .unwrap();

            Ok(Self {
                env: env.to_object(py),
                render_view,
                gymnasium: gymnasium.into(),
                gym_env,
            })
        })
    }

    fn set_render_view(&mut self, render_view: String) -> PyResult<()> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("render_mode", &render_view)?;
            let gymnasium = self.gymnasium.bind(py);

            let new_env = gymnasium.call_method("make", (&self.gym_env,), Some(&kwargs))?;

            self.env = new_env.to_object(py);
            self.render_view = render_view;

            Ok(())
        })
    }

    pub fn reset(&self) -> PyResult<State> {
        Python::with_gil(|py| {
            let env_bound = self.env.bind(py);
            let result_tuple = env_bound.call_method0("reset")?;

            let observation: Vec<f32> = result_tuple.get_item(0)?.extract()?;

            Ok(State {
                cart_position: observation[0],
                cart_velocity: observation[1],
                pole_angle: observation[2],
                pole_velocity: observation[3],
            })
        })
    }

    pub fn step(&self, action: Action) -> PyResult<(State, f32, bool)> {
        Python::with_gil(|py| {
            let action_int = match action {
                Action::Left => 0,
                Action::Right => 1,
            };

            let env_bound = self.env.bind(py);
            let result_tuple = env_bound.call_method1("step", (action_int,))?;

            let observation: Vec<f32> = result_tuple.get_item(0)?.extract()?;
            let reward: f32 = result_tuple.get_item(1)?.extract()?;
            let terminated: bool = result_tuple.get_item(2)?.extract()?;
            let truncated: bool = result_tuple.get_item(3)?.extract()?;

            let next_state = State {
                cart_position: observation[0],
                cart_velocity: observation[1],
                pole_angle: observation[2],
                pole_velocity: observation[3],
            };

            let done = terminated || truncated;

            Ok((next_state, reward, done))
        })
    }

    pub fn render(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            self.env.bind(py).call_method0("render")?;
            Ok(())
        })
    }

    pub fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            self.env.bind(py).call_method0("close")?;
            Ok(())
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut cartpole = GymnasiumWrapper::new("rgb_array".to_string())?;
    let mut dqn_agent = DQNAgent::new();
    let batch_size = 64;
    let episodes = 500;
    dqn_agent.epsilon_decay = 0.995;
    let to_train = false;
    dqn_agent.epsilon = 1.00;

    let model = "alpha.ot";

    if std::path::Path::new(model).exists() {
        println!("Loading saved agent from {}...", model);
        dqn_agent.load(model)?;
    }

    if to_train {
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

                if done {
                    break;
                }
            }

            if episode % 10 == 0 {
                dqn_agent.update_target_network();
                println!(
                    "Episode: {}, Total Reward: {:.2}, Epsilon: {:.3}",
                    episode, total_reward, dqn_agent.epsilon
                )
            }
        }

        println!("訓練終了！エージェントを保存します。。。");
        dqn_agent.save(model)?;
        println!("保存しました！");
    } else {
        dqn_agent.epsilon = 0.00;
        let mut state = cartpole.reset()?;
        let mut total_reward = 0.0;

        cartpole.set_render_view("human".to_string())?;

        for game in 0..15 {
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
