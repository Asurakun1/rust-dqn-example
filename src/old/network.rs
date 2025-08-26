use rand::Rng;
use tch::{
    nn::{self, Module, Optimizer, OptimizerConfig},
    Device, Kind, Tensor,
};

use crate::qnet::{Experience, QNetwork};
pub const LEARN_RATE: f64 = 1e-2;
pub const EXPLORATION_DECAY: f64 = 0.999;
pub const NEURONS: i64 = 64;
pub const EPSILON_CAP: f64 = 0.01;

const EPSILON: f64 = 1.0;
// DqnAgent struct remains the same
pub struct DqnAgent {
    q_network: QNetwork,
    target_network: QNetwork,
    pub var_store: nn::VarStore,
    optimizer: Optimizer,
    pub device: Device,
    gamma: f64,
    pub epsilon: f64,
    target_update_steps: usize,
    steps_done: usize,
}

impl DqnAgent {
    pub fn new(state_dim: i64, action_dim: i64) -> Self {
        let device = Device::cuda_if_available();
        let var_store = nn::VarStore::new(device);
        let q_network = QNetwork::new(&var_store, state_dim, action_dim);
        let mut target_network = QNetwork::new(&var_store, state_dim, action_dim);

        tch::no_grad(|| {
            target_network.fc1.ws.copy_(&q_network.fc1.ws);
            target_network.fc2.ws.copy_(&q_network.fc2.ws);
            target_network.fc3.ws.copy_(&q_network.fc3.ws);
        });

        // The .build() method is now available because OptimizerConfig is in scope.
        let optimizer = nn::Adam::default().build(&var_store, LEARN_RATE).unwrap();

        Self {
            q_network,
            target_network,
            var_store,
            optimizer,
            device,
            gamma: 0.99,
            epsilon: EPSILON,
            target_update_steps: 10,
            steps_done: 0,
        }
    }

    pub fn select_action(&mut self, state: &Tensor) -> i64 {
        let mut rng = rand::thread_rng();
        self.steps_done += 1;
        self.epsilon = f64::max(EPSILON_CAP, self.epsilon * EXPLORATION_DECAY);

        if rng.gen::<f64>() < self.epsilon {
            let action_space_size = self.q_network.fc3.ws.size()[0];
            rng.gen_range(0..action_space_size)
        } else {
            let q_values = tch::no_grad(|| self.q_network.forward(state));
            q_values.argmax(None, false).int64_value(&[])
        }
    }

    pub fn learn(&mut self, experiences: Vec<Experience>) {
        // --- FIX 3: Manually de-structure the experiences ---
        // .unzip() doesn't work for 5-element tuples, so we do it manually.
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut dones = Vec::new();

        for (s, a, r, ns, d) in experiences {
            states.push(s);
            actions.push(a);
            rewards.push(r);
            next_states.push(ns);
            dones.push(d);
        }

        let states = Tensor::cat(&states, 0).to(self.device);
        let actions = Tensor::from_slice(&actions).unsqueeze(1).to(self.device);
        let rewards = Tensor::from_slice(&rewards)
            .to_kind(Kind::Float)
            .to(self.device);
        let next_states = Tensor::cat(&next_states, 0).to(self.device);
        let dones = Tensor::from_slice(
            &dones
                .iter()
                .map(|&d| if d { 0.0 } else { 1.0 })
                .collect::<Vec<f32>>(),
        )
        .to(self.device);

        let next_q_values = tch::no_grad(|| self.target_network.forward(&next_states));
        let next_max_q = next_q_values.max_dim(1, false).0;
        let expected_q_values = rewards + (self.gamma * next_max_q * dones);

        let predicted_q_values = self.q_network.forward(&states).gather(1, &actions, false);
        let loss = predicted_q_values.smooth_l1_loss(
            &expected_q_values.unsqueeze(1),
            tch::Reduction::Mean,
            1.0,
        );

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        if self.steps_done % self.target_update_steps == 0 {
            tch::no_grad(|| {
                self.target_network.fc1.ws.copy_(&self.q_network.fc1.ws);
                self.target_network.fc2.ws.copy_(&self.q_network.fc2.ws);
                self.target_network.fc3.ws.copy_(&self.q_network.fc3.ws);
            });
        }
    }
}
