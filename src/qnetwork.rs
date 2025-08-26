use rand::{random_range, seq::index};
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
            .add(nn::linear(vs / "layer2", 64, 2, Default::default()));
        Self { seq }
    }
}

impl nn::Module for QNetwork {
    //The Forward Pass
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.seq.forward(xs)
    }
}

pub struct Transition {
    state: State,
    action: Action,
    reward: f32,
    next_state: State,
    done: bool,
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
        if self.capacity > self.transitions.len() {
            self.transitions.pop_front();
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        if self.transitions.len() < batch_size {
            let what_we_have = self.transitions.iter().collect();
            what_we_have
        } else {
            let mut rng = rand::rng();
            let indices = index::sample(&mut rng, self.transitions.len(), batch_size);

            let batch = indices
                .into_iter()
                .map(|index| &self.transitions[index])
                .collect();
            batch
        }
    }
}

pub struct DQNAgent {
    q_network: QNetwork,
    target_network: QNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: tch::nn::Optimizer,
    step_counter: u64,
    epsilon: f32,
    gamma: f32,
    vs_main: VarStore,
    vs_target: VarStore,
}

impl DQNAgent {
    pub fn new() -> Self {
        let device = Device::cuda_if_available();
        let vs_main = VarStore::new(device);
        let q_network = QNetwork::new(&vs_main.root());

        let mut vs_target = VarStore::new(device);
        let target_network = QNetwork::new(&vs_target.root());

        vs_target.copy(&vs_main).unwrap();
        let optimizer = tch::nn::Adam::default().build(&vs_main, 1e-3).unwrap();

        let replay_buffer = ReplayBuffer::new(10000);
        let step_counter = 0;
        let epsilon = 1.0;
        let gamma = 0.99;

        Self {
            q_network,
            target_network,
            replay_buffer,
            optimizer,
            step_counter,
            epsilon,
            gamma,
            vs_main,
            vs_target,
        }
    }

    pub fn action(&self, state: &State) -> Action {
        let rand: f32 = random_range(0.0..1.0);

        if self.epsilon < rand {
            let random_decision = random_range(0..1);
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
        The Formula
         */

        let next_q_values = self.target_network.forward(&tensor_next_states);
        let max_next_q_values = next_q_values.max_dim(1, false).0;

        let target_q_values =
            &tensor_rewards + (self.gamma * max_next_q_values.view([-1, 1]) * &tensor_dones);
    }
}
