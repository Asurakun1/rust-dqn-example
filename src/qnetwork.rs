use rand::{random_range, seq::IteratorRandom};
use std::collections::VecDeque;
use tch::{
    nn::{self, Module, OptimizerConfig, VarStore},
    Device, Tensor,
};

use crate::environment::{Action, State};

#[derive(Debug)]
pub struct QNetwork {
    pub seq: nn::Sequential,
}

impl QNetwork {
    pub fn new(vs: &nn::Path) -> Self {
        let seq = nn::seq()
            .add(nn::linear(vs / "layer1", 4, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer2", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer3", 64, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer4", 64, 2, Default::default()));
        Self { seq }
    }
}

impl nn::Module for QNetwork {
    //The Forward Pass
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.seq.forward(xs)
    }
}

#[derive(Clone, Copy)]
pub struct Transition {
    pub state: State,
    pub action: Action,
    pub reward: f32,
    pub next_state: State,
    pub done: bool,
}

pub struct ReplayBuffer {
    transitions: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len() > self.capacity {
            self.transitions.pop_front();
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::rng();
        let sample_size = std::cmp::min(batch_size, self.transitions.len());
        self.transitions.iter().choose_multiple(&mut rng, sample_size)
    }
}

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub struct AgentConfig {
    pub learning_rate: f64,
    pub gamma: f32,
    pub epsilon_decay: f32,
    pub epsilon_min: f32,
    pub replay_buffer_capacity: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct AgentState {
    epsilon: f32,
    model_bytes: Vec<u8>,
}

pub struct DQNAgent {
    q_network: QNetwork,
    target_network: QNetwork,
    pub replay_buffer: ReplayBuffer,
    optimizer: tch::nn::Optimizer,
    pub step_counter: u64,
    pub epsilon: f32,
    gamma: f32,
    vs_main: VarStore,
    vs_target: VarStore,
    pub epsilon_decay: f32,
    epsilon_min: f32,
}

impl DQNAgent {
    pub fn new(config: AgentConfig) -> Self {
        let device = Device::cuda_if_available();
        let vs_main = VarStore::new(device);
        let q_network = QNetwork::new(&vs_main.root());

        let mut vs_target = VarStore::new(device);
        let target_network = QNetwork::new(&vs_target.root());

        vs_target.copy(&vs_main).unwrap();

        let optimizer = tch::nn::Adam::default()
            .build(&vs_main, config.learning_rate)
            .unwrap();

        let replay_buffer = ReplayBuffer::new(config.replay_buffer_capacity);
        let step_counter = 0;
        let epsilon = 1.0;

        Self {
            q_network,
            target_network,
            replay_buffer,
            optimizer,
            step_counter,
            epsilon,
            gamma: config.gamma,
            vs_main,
            vs_target,
            epsilon_decay: config.epsilon_decay,
            epsilon_min: config.epsilon_min,
        }
    }

    pub fn action(&self, state: &State) -> Action {
        let rand: f32 = random_range(0.0..1.0);

        if self.epsilon > rand {
            let random_decision = random_range(0..2);
            match random_decision {
                0 => {
                    // go left
                    return Action::Left;
                }
                _ => {
                    // go right
                    return Action::Right;
                }
            }
        } else {
            /*
            Convert to tensors
             */
            let state_slices = [
                state.cart_position,
                state.cart_velocity,
                state.pole_angle,
                state.pole_velocity,
            ];

            let state_tensor = Tensor::from_slice(&state_slices)
                .to(self.vs_main.device())
                .view([1, 4]);

            let q_values = self.q_network.forward(&state_tensor);

            let best_action_index = q_values.argmax(1, false).int64_value(&[]);

            if best_action_index == 0 {
                return Action::Left;
            } else {
                return Action::Right;
            }
        }
    }

    pub fn train(&mut self, batch_size: usize) {
        if self.replay_buffer.transitions.len() < batch_size {
            return;
        }

        /*
        Preparation
         */
        let training_sample = self.replay_buffer.sample(batch_size);

        let states: Vec<State> = training_sample
            .iter()
            .map(|transition| transition.state.clone())
            .collect();

        let actions: Vec<Action> = training_sample
            .iter()
            .map(|transition| transition.action.clone())
            .collect();

        let rewards: Vec<f32> = training_sample
            .iter()
            .map(|transition| transition.reward)
            .collect();

        let next_states: Vec<State> = training_sample
            .iter()
            .map(|transition| transition.next_state.clone())
            .collect();

        let dones: Vec<bool> = training_sample
            .iter()
            .map(|transition| transition.done)
            .collect();

        /*
        Tensor Conversion
         */
        let mut flat_states = Vec::with_capacity(states.len() * 4);

        for state in states {
            flat_states.push(state.cart_position);
            flat_states.push(state.cart_velocity);
            flat_states.push(state.pole_angle);
            flat_states.push(state.pole_velocity);
        }

        let tensor_states = Tensor::from_slice(&flat_states)
            .to(self.vs_main.device())
            .view([batch_size as i64, 4]);

        let mut flat_next_states = Vec::with_capacity(next_states.len() * 4);

        for next_state in next_states {
            flat_next_states.push(next_state.cart_position);
            flat_next_states.push(next_state.cart_velocity);
            flat_next_states.push(next_state.pole_angle);
            flat_next_states.push(next_state.pole_velocity);
        }

        let tensor_next_states = Tensor::from_slice(&flat_next_states)
            .to(self.vs_main.device())
            .view([batch_size as i64, 4]);

        let actions_to_i64: Vec<i64> = actions
            .iter()
            .map(|act| match act {
                Action::Left => 0,
                Action::Right => 1,
            })
            .collect();

        let tensor_actions = Tensor::from_slice(&actions_to_i64)
            .to(self.vs_main.device())
            .view([-1, 1]);

        let tensor_rewards = Tensor::from_slice(&rewards)
            .to(self.vs_main.device())
            .view([-1, 1]);

        let dones_f32: Vec<f32> = dones
            .iter()
            .map(|done| if *done { 0.0 } else { 1.0 })
            .collect();

        let tensor_dones = Tensor::from_slice(&dones_f32)
            .to(self.vs_main.device())
            .view([-1, 1]);

        /*
        End of Tensor Conversion
         */

        /*
        The Formula Double DQN
         */

        let next_actions = self.q_network.forward(&tensor_next_states).argmax(1, true);

        let max_next_q_values = self
            .target_network
            .forward(&tensor_next_states)
            .gather(1, &next_actions, false)
            .squeeze_dim(1);

        let target_q_values =
            &tensor_rewards + (self.gamma * max_next_q_values.view([-1, 1]) * &tensor_dones);

        let predicted_q_values =
            self.q_network
                .forward(&tensor_states)
                .gather(1, &tensor_actions, false);

        let loss = predicted_q_values.mse_loss(&target_q_values, tch::Reduction::Mean);

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        self.step_counter += 1;

        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
    }

    pub fn update_target_network(&mut self) {
        self.vs_target.copy(&self.vs_main).unwrap();
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer: Vec<u8> = Vec::new();
        self.vs_main.save_to_stream(&mut buffer)?;

        let state = AgentState {
            epsilon: self.epsilon,
            model_bytes: buffer,
        };

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &state)?;

        Ok(())
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let state: AgentState = bincode::deserialize_from(&mut reader)?;

        self.epsilon = state.epsilon;
        let mut model_reader = std::io::Cursor::new(state.model_bytes);
        self.vs_main.load_from_stream(&mut model_reader)?;
        self.vs_target.copy(&self.vs_main)?;

        Ok(())
    }
}
