use std::collections::VecDeque;

use rand::seq::SliceRandom;
use tch::{nn, Tensor};

use crate::network::NEURONS;

#[derive(Debug)]
pub struct QNetwork {
    pub(crate) fc1: nn::Linear,
    pub(crate) fc2: nn::Linear,
    pub(crate) fc3: nn::Linear,
}

impl QNetwork {
    pub fn new(vs: &nn::VarStore, in_dim: i64, out_dim: i64) -> Self {
        let hidden_dim = NEURONS;
        let vs_path = &vs.root();
        Self {
            fc1: nn::linear(vs_path, in_dim, hidden_dim, Default::default()),
            fc2: nn::linear(vs_path, hidden_dim, hidden_dim, Default::default()),
            fc3: nn::linear(vs_path, hidden_dim, out_dim, Default::default()),
        }
    }
}

impl nn::Module for QNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
    }
}

// Experience tuple remains the same
pub type Experience = (Tensor, i64, f64, Tensor, bool);

// ReplayBuffer struct remains the same
#[derive(Debug)]
pub struct ReplayBuffer {
    memory: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            memory: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Experience) {
        if self.memory.len() == self.capacity {
            self.memory.pop_front();
        }
        self.memory.push_back(experience);
    }

    // --- FIX 2: Correctly sample and copy Tensors ---
    pub fn sample(&self, batch_size: usize) -> Option<Vec<Experience>> {
        if self.memory.len() < batch_size {
            return None;
        }
        let mut rng = rand::thread_rng();
        // choose_multiple gives us an iterator of references.
        // We map over them and manually copy the tensors to create new owned experiences.
        let samples = self
            .memory
            .as_slices()
            .0
            .choose_multiple(&mut rng, batch_size)
            .map(|(s, a, r, ns, d)| (s.copy(), *a, *r, ns.copy(), *d))
            .collect();
        Some(samples)
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }
}
