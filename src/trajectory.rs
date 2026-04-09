use std::{rc::Rc, sync::Arc, hash::Hash};

use crate::{
    observation::Observation,
    observer::{Observer},
};

pub struct Trajectory {
    pub id: u64,
    pub observations: Vec<Observation>,
}
