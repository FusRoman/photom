use crate::observation::Observation;

pub struct Trajectory {
    pub id: u64,
    pub observations: Vec<Observation>,
}
