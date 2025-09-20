use pyo3::{prelude::*, types::PyDict};
use crate::environment::{Action, State};

pub struct GymnasiumWrapper {
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

    pub fn set_render_view(&mut self, render_view: String) -> PyResult<()> {
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
