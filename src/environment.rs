const CART_DISTANCE: f32 = 2.4;
const CART_MAX_ANGLE: f32 = 0.209;

#[derive(Debug)]
pub struct CartPole {
    current_state: State,
    gravity: f32,
    force_magnitude: f32,
    time_step: f32,
    pole_length: f32,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct State {
    pub cart_position: f32,
    pub cart_velocity: f32,
    pub pole_angle: f32,
    pub pole_velocity: f32,
}

#[derive(Clone, Copy)]
pub enum Action {
    Left,
    Right,
}

impl CartPole {
    pub fn new() -> Self {
        Self {
            current_state: State::default(),
            gravity: 9.8,
            force_magnitude: 10.0,
            time_step: 0.02,
            pole_length: 0.5,
        }
    }

    pub fn step(&mut self, action: Action) -> (State, f32, bool) {
        let force = match action {
            Action::Left => -self.force_magnitude,
            Action::Right => self.force_magnitude,
        };

        let (theta, theta_dot) = (
            self.current_state.pole_angle,
            self.current_state.pole_velocity,
        );

        let pole_mass = 0.1;
        let cart_mass = 1.0;

        let total_mass = pole_mass + cart_mass;

        /*
        angular acceleration theta_acc
         */

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let temp =
            (force + self.pole_length * pole_mass * theta_dot.powi(2) * sin_theta) / total_mass;

        // Angular Acceleration = Net Torque / Moment of Inertia
        let theta_acc = (self.gravity * sin_theta - temp * cos_theta)
            / (self.pole_length * (4.0 / 3.0 - pole_mass * cos_theta.powi(2) / total_mass));

        // cart acceleration
        let x_acc = temp - self.pole_length * pole_mass * theta_acc * cos_theta / total_mass;

        //update the state
        self.current_state.cart_velocity =
            self.current_state.cart_velocity + x_acc * self.time_step;
        self.current_state.cart_position =
            self.current_state.cart_position + self.current_state.cart_velocity * self.time_step;

        self.current_state.pole_velocity =
            self.current_state.pole_velocity + theta_acc * self.time_step;
        self.current_state.pole_angle =
            self.current_state.pole_angle + self.current_state.pole_velocity * self.time_step;

        let done = self.current_state.cart_position < -CART_DISTANCE
            || self.current_state.cart_position > CART_DISTANCE
            || self.current_state.pole_angle < -CART_MAX_ANGLE
            || self.current_state.pole_angle > CART_MAX_ANGLE;

        let reward = if done { 0.0 } else { 1.0 };

        (self.current_state.clone(), reward, done)
    }

    pub fn reset(&mut self) -> State {
        self.current_state = State::default();
        self.current_state.clone()
    }
}
