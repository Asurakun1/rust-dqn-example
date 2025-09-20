//! This module defines the CartPole environment, including its state representation,
//! possible actions, and the physics simulation for stepping through the environment.

// Constants defining the limits of the CartPole environment.
// These are used to determine when an episode terminates.
const CART_DISTANCE: f32 = 2.4; // Maximum distance the cart can travel from the center.
const CART_MAX_ANGLE: f32 = 0.209; // Maximum angle the pole can deviate from vertical (radians).

/// Represents the CartPole environment.
/// It holds the current state of the cart and pole, and defines the physics of the environment.
#[derive(Debug)]
pub struct CartPole {
    current_state: State,
    gravity: f32,
    force_magnitude: f32,
    time_step: f32,
    pole_length: f32,
}

/// Represents the state of the CartPole environment at a given time.
/// Contains the position and velocity of the cart, and the angle and angular velocity of the pole.
#[derive(Default, Debug, Clone, Copy)]
pub struct State {
    pub cart_position: f32,
    pub cart_velocity: f32,
    pub pole_angle: f32,
    pub pole_velocity: f32,
}

/// Defines the possible actions the agent can take in the CartPole environment.
#[derive(Clone, Copy)]
pub enum Action {
    Left,
    Right,
}

impl Default for CartPole {
    /// Provides a default constructor for CartPole, initializing it with standard parameters.
    fn default() -> Self {
        Self::new()
    }
}

impl CartPole {
    /// Creates a new CartPole environment with default initial parameters.
    pub fn new() -> Self {
        Self {
            current_state: State::default(),
            gravity: 9.8, // Gravity constant (m/s^2)
            force_magnitude: 10.0, // Force applied to the cart when an action is taken
            time_step: 0.02, // Duration of each simulation step (seconds)
            pole_length: 0.5, // Length of the pole (meters)
        }
    }

    /// Advances the environment by one step given an action.
    /// It applies the force, updates the cart and pole dynamics, and checks for termination conditions.
    /// Returns the new state, the reward, and a boolean indicating if the episode is done.
    pub fn step(&mut self, action: Action) -> (State, f32, bool) {
        // Determine the force to apply based on the chosen action.
        let force = match action {
            Action::Left => -self.force_magnitude,
            Action::Right => self.force_magnitude,
        };

        // Extract current pole angle and angular velocity for physics calculations.
        let (theta, theta_dot) = (
            self.current_state.pole_angle,
            self.current_state.pole_velocity,
        );

        // Define masses for the pole and cart.
        let pole_mass = 0.1;
        let cart_mass = 1.0;

        let total_mass = pole_mass + cart_mass;

        // Calculate angular acceleration of the pole.
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let temp =
            (force + self.pole_length * pole_mass * theta_dot.powi(2) * sin_theta) / total_mass;

        // Angular Acceleration = Net Torque / Moment of Inertia
        let theta_acc = (self.gravity * sin_theta - temp * cos_theta)
            / (self.pole_length * (4.0 / 3.0 - pole_mass * cos_theta.powi(2) / total_mass));

        // Calculate acceleration of the cart.
        let x_acc = temp - self.pole_length * pole_mass * theta_acc * cos_theta / total_mass;

        // Update cart position and velocity based on calculated accelerations.
        self.current_state.cart_velocity += x_acc * self.time_step;
        self.current_state.cart_position += self.current_state.cart_velocity * self.time_step;

        // Update pole angle and angular velocity based on calculated accelerations.
        self.current_state.pole_velocity += theta_acc * self.time_step;
        self.current_state.pole_angle += self.current_state.pole_velocity * self.time_step;

        // Determine if the episode is done based on cart position or pole angle limits.
        let done = self.current_state.cart_position < -CART_DISTANCE
            || self.current_state.cart_position > CART_DISTANCE
            || self.current_state.pole_angle < -CART_MAX_ANGLE
            || self.current_state.pole_angle > CART_MAX_ANGLE;

        // Assign reward: 1.0 if not done, 0.0 if done (standard for CartPole).
        let reward = if done { 0.0 } else { 1.0 };

        (self.current_state, reward, done)
    }

    /// Resets the environment to its initial state.
    /// The cart is at the center, and the pole is upright with no initial velocity.
    /// Returns the initial state.
    pub fn reset(&mut self) -> State {
        self.current_state = State::default();
        self.current_state
    }
}
