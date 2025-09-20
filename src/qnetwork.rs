//! This module defines the core components of the Deep Q-Network (DQN) agent.
//! It includes the Q-Network architecture, Replay Buffer for experience replay,
//! and the DQNAgent itself, which handles action selection, training, and model persistence.

use rand::{random_range, seq::IteratorRandom};
use std::collections::VecDeque;
use tch::{
    nn::{self, Module, OptimizerConfig, VarStore},
    Device, Tensor,
};

use crate::environment::{Action, State};

/// Represents the Q-Network, a neural network that approximates Q-values.
/// It takes a state as input and outputs Q-values for each possible action.
#[derive(Debug)]
pub struct QNetwork {
    pub seq: nn::Sequential,
}

impl QNetwork {
    /// Creates a new Q-Network with a sequential architecture.
    /// The network consists of multiple linear layers with ReLU activations.
    pub fn new(vs: &nn::Path) -> Self {
        let seq = nn::seq()
            .add(nn::linear(vs / "layer1", 4, 64, Default::default())) // Input layer (4 features: cart_position, cart_velocity, pole_angle, pole_velocity)
            .add_fn(|xs| xs.relu()) // ReLU activation
            .add(nn::linear(vs / "layer2", 64, 64, Default::default())) // Hidden layer
            .add_fn(|xs| xs.relu()) // ReLU activation
            .add(nn::linear(vs / "layer3", 64, 64, Default::default())) // Hidden layer
            .add_fn(|xs| xs.relu()) // ReLU activation
            .add(nn::linear(vs / "layer4", 64, 2, Default::default())); // Output layer (2 actions: Left, Right)
        Self { seq }
    }
}

impl nn::Module for QNetwork {
    /// Implements the forward pass for the Q-Network.
    /// Takes an input tensor (state) and returns the output tensor (Q-values).
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.seq.forward(xs)
    }
}

/// Represents a single transition (experience) in the environment.
/// Used for storing and replaying past experiences to train the agent.
#[derive(Clone, Copy)]
pub struct Transition {
    pub state: State,
    pub action: Action,
    pub reward: f32,
    pub next_state: State,
    pub done: bool,
}

/// Implements a Replay Buffer for storing and sampling experiences.
/// This is crucial for stabilizing DQN training by breaking correlations in the observation sequence.
pub struct ReplayBuffer {
    transitions: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    /// Creates a new ReplayBuffer with a specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds a new transition to the replay buffer.
    /// If the buffer exceeds its capacity, the oldest transition is removed.
    pub fn push(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len() > self.capacity {
            self.transitions.pop_front();
        }
    }

    /// Samples a random batch of transitions from the replay buffer.
    /// Used for training the Q-Network.
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::rng();
        let sample_size = std::cmp::min(batch_size, self.transitions.len());
        self.transitions.iter().choose_multiple(&mut rng, sample_size)
    }
}

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Configuration parameters for the DQNAgent.
/// These hyperparameters control the agent's learning behavior.
pub struct AgentConfig {
    pub learning_rate: f64,
    pub gamma: f32,
    pub epsilon_decay: f32,
    pub epsilon_min: f32,
    pub replay_buffer_capacity: usize,
}

/// Represents the serializable state of the DQNAgent for saving and loading.
/// Contains the current epsilon value and the Q-network's weights.
#[derive(Serialize, Deserialize, Debug)]
struct AgentState {
    epsilon: f32,
    model_bytes: Vec<u8>,
}

/// Implements the Deep Q-Network (DQN) agent.
/// Manages the Q-network, target network, replay buffer, and training process.
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
    /// Creates a new DQNAgent with the given configuration.
    /// Initializes the Q-network, target network, optimizer, and replay buffer.
    pub fn new(config: AgentConfig) -> Self {
        let device = Device::cuda_if_available();
        let vs_main = VarStore::new(device);
        let q_network = QNetwork::new(&vs_main.root());

        let mut vs_target = VarStore::new(device);
        let target_network = QNetwork::new(&vs_target.root());

        // Copy initial weights from Q-network to target network.
        vs_target.copy(&vs_main).unwrap();

        // Initialize the Adam optimizer with the specified learning rate.
        let optimizer = tch::nn::Adam::default()
            .build(&vs_main, config.learning_rate)
            .unwrap();

        // Initialize the replay buffer with the specified capacity.
        let replay_buffer = ReplayBuffer::new(config.replay_buffer_capacity);
        let step_counter = 0;
        let epsilon = 1.0; // Initial exploration rate

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

    /// Selects an action based on the current state using an epsilon-greedy policy.
    /// During exploration (epsilon > random), a random action is chosen.
    /// Otherwise, the action with the highest Q-value from the Q-network is chosen.
    pub fn action(&self, state: &State) -> Action {
        let rand: f32 = random_range(0.0..1.0);

        if self.epsilon > rand {
            // Explore: choose a random action (0 for Left, 1 for Right).
            let random_decision = random_range(0..2);
            match random_decision {
                0 => {
                    Action::Left
                }
                _ => {
                    Action::Right
                }
            }
        } else {
            // Exploit: choose the action with the highest Q-value from the Q-network.
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
                Action::Left
            } else {
                Action::Right
            }
        }
    }

    /// Trains the Q-network using a batch of experiences sampled from the replay buffer.
    /// Implements the Double DQN update rule.
    pub fn train(&mut self, batch_size: usize) {
        // Ensure there are enough transitions in the replay buffer to sample a batch.
        if self.replay_buffer.transitions.len() < batch_size {
            return;
        }

        // Sample a batch of transitions from the replay buffer.
        let training_sample = self.replay_buffer.sample(batch_size);

        // Extract individual components (states, actions, rewards, next_states, dones) from the sampled transitions.
        let states: Vec<State> = training_sample
            .iter()
            .map(|transition| transition.state)
            .collect();

        let actions: Vec<Action> = training_sample
            .iter()
            .map(|transition| transition.action)
            .collect();

        let rewards: Vec<f32> = training_sample
            .iter()
            .map(|transition| transition.reward)
            .collect();

        let next_states: Vec<State> = training_sample
            .iter()
            .map(|transition| transition.next_state)
            .collect();

        let dones: Vec<bool> = training_sample
            .iter()
            .map(|transition| transition.done)
            .collect();

        // Convert extracted data into Tensors for neural network processing.
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

        // Calculate target Q-values using the Double DQN formula.
        // Q_target = reward + gamma * max_a(Q_target(next_state, a))
        // For terminal states (done), Q_target is just the reward.
        let next_actions = self.q_network.forward(&tensor_next_states).argmax(1, true);

        let max_next_q_values = self
            .target_network
            .forward(&tensor_next_states)
            .gather(1, &next_actions, false)
            .squeeze_dim(1);

        let target_q_values =
            &tensor_rewards + (self.gamma * max_next_q_values.view([-1, 1]) * &tensor_dones);

        // Get predicted Q-values from the main Q-network for the taken actions.
        let predicted_q_values =
            self.q_network
                .forward(&tensor_states)
                .gather(1, &tensor_actions, false);

        // Calculate the Mean Squared Error loss between predicted and target Q-values.
        let loss = predicted_q_values.mse_loss(&target_q_values, tch::Reduction::Mean);

        // Perform backpropagation and update Q-network weights.
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        // Increment the step counter.
        self.step_counter += 1;

        // Decay epsilon (exploration rate) if it's above the minimum threshold.
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
    }

    /// Updates the target network by copying the weights from the main Q-network.
    /// This is done periodically to stabilize training.
    pub fn update_target_network(&mut self) {
        self.vs_target.copy(&self.vs_main).unwrap();
    }

    /// Saves the current state of the DQNAgent (epsilon and Q-network weights) to a file.
    /// The model is serialized using bincode.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer: Vec<u8> = Vec::new();
        // Save the main Q-network's weights to a byte buffer.
        self.vs_main.save_to_stream(&mut buffer)?;

        // Create an AgentState struct to hold epsilon and model bytes.
        let state = AgentState {
            epsilon: self.epsilon,
            model_bytes: buffer,
        };

        // Serialize the AgentState to a file using bincode.
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &state)?;

        Ok(())
    }

    /// Loads the state of the DQNAgent (epsilon and Q-network weights) from a file.
    /// The model is deserialized using bincode.
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Open and deserialize the AgentState from the file.
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let state: AgentState = bincode::deserialize_from(&mut reader)?;

        // Restore epsilon and load the main Q-network's weights from the buffer.
        self.epsilon = state.epsilon;
        let mut model_reader = std::io::Cursor::new(state.model_bytes);
        self.vs_main.load_from_stream(&mut model_reader)?;
        // Copy the loaded weights to the target network.
        self.vs_target.copy(&self.vs_main)?;

        Ok(())
    }
}
