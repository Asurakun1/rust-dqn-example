//! This module provides a Rust wrapper for interacting with Python's Gymnasium environments.
//! It uses `pyo3` to bridge between Rust and Python, allowing the Rust DQN agent to
//! interact with standard reinforcement learning environments like CartPole.

use pyo3::{prelude::*, types::PyDict};
use crate::environment::{Action, State};

/// A wrapper struct that encapsulates a Python Gymnasium environment.
/// It handles the necessary Python interoperability to reset, step, render, and close the environment.
pub struct GymnasiumWrapper {
    env: PyObject,      // The Python Gymnasium environment object.
    render_view: String, // The current render mode (e.g., "rgb_array", "human").
    gymnasium: PyObject, // The Python 'gymnasium' module object.
    gym_env: String,     // The name of the Gymnasium environment (e.g., "CartPole-v1").
}

impl GymnasiumWrapper {
    /// Creates a new `GymnasiumWrapper` instance, initializing the specified Gymnasium environment.
    /// `render_view` determines the rendering mode of the environment.
    pub fn new(render_view: String) -> PyResult<Self> {
        let gym_env = "CartPole-v1".to_string(); // Currently hardcoded to CartPole-v1.
        Python::with_gil(|py| {
            // Import the 'gymnasium' Python module.
            let gymnasium = py.import_bound("gymnasium")?;

            // Prepare keyword arguments for the 'make' method (e.g., render_mode).
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("render_mode", render_view.as_str())?;
            // Create the environment instance using gymnasium.make().
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

    /// Sets the rendering mode for the Gymnasium environment.
    /// This can be used to switch between modes like "human" (for display) or "rgb_array" (for data).
    pub fn set_render_view(&mut self, render_view: String) -> PyResult<()> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("render_mode", &render_view)?;
            let gymnasium = self.gymnasium.bind(py);

            // Re-create the environment with the new render mode.
            let new_env = gymnasium.call_method("make", (&self.gym_env,), Some(&kwargs))?;

            self.env = new_env.to_object(py);
            self.render_view = render_view;

            Ok(())
        })
    }

    /// Resets the Gymnasium environment to its initial state.
    /// Returns the initial observation (state) from the environment.
    pub fn reset(&self) -> PyResult<State> {
        Python::with_gil(|py| {
            // Call the 'reset()' method on the Python environment object.
            let env_bound = self.env.bind(py);
            let result_tuple = env_bound.call_method0("reset")?;

            // Extract the observation (state) from the result tuple.
            let observation: Vec<f32> = result_tuple.get_item(0)?.extract()?;

            // Convert the observation vector into the Rust `State` struct.
            Ok(State {
                cart_position: observation[0],
                cart_velocity: observation[1],
                pole_angle: observation[2],
                pole_velocity: observation[3],
            })
        })
    }

    /// Performs a step in the Gymnasium environment with the given action.
    /// Returns the new state, reward, and a boolean indicating if the episode is done.
    pub fn step(&self, action: Action) -> PyResult<(State, f32, bool)> {
        Python::with_gil(|py| {
            // Convert the Rust `Action` enum to an integer for the Python environment.
            let action_int = match action {
                Action::Left => 0,
                Action::Right => 1,
            };

            // Call the 'step()' method on the Python environment object.
            let env_bound = self.env.bind(py);
            let result_tuple = env_bound.call_method1("step", (action_int,))?;

            // Extract observation, reward, terminated, and truncated flags from the result tuple.
            let observation: Vec<f32> = result_tuple.get_item(0)?.extract()?;
            let reward: f32 = result_tuple.get_item(1)?.extract()?;
            let terminated: bool = result_tuple.get_item(2)?.extract()?;
            let truncated: bool = result_tuple.get_item(3)?.extract()?;

            // Convert the observation vector into the Rust `State` struct.
            let next_state = State {
                cart_position: observation[0],
                cart_velocity: observation[1],
                pole_angle: observation[2],
                pole_velocity: observation[3],
            };

            // Determine if the episode is done (either terminated or truncated).
            let done = terminated || truncated;

            Ok((next_state, reward, done))
        })
    }

    /// Renders the Gymnasium environment.
    /// This typically opens a window to display the environment visually.
    pub fn render(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            self.env.bind(py).call_method0("render")?;
            Ok(())
        })
    }

    /// Closes the Gymnasium environment, releasing any resources (e.g., rendering window).
    pub fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            self.env.bind(py).call_method0("close")?;
            Ok(())
        })
    }
}
