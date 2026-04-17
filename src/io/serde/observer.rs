// ---------------------------------------------------------------------------
// Serde support for ObserverDataset
// ---------------------------------------------------------------------------
//
// `ObserverDataset` cannot use a derived `Serialize`/`Deserialize` because the
// `mpc_observers` field is a `OnceLock` wrapping a network-fetched cache.
// We therefore use a lightweight proxy struct that captures only the two
// *persistent* fields and reconstruct `ObserverDataset` on deserialisation by
// calling `ObserverDataset::new`, which initialises a fresh (empty) `OnceLock`.

use crate::observer::{Observer, dataset::ObserverDataset, error_model::ObsErrorModel};

/// Serialisable proxy for [`ObserverDataset`].
///
/// Only the two persistent fields are stored:
/// - `custom_observers` — geodetic observer sites supplied by the caller.
/// - `mpc_error_model`  — astrometric error-model variant used when the MPC
///   catalogue is lazily loaded.
///
/// The MPC network cache (`mpc_observers`) is **not** serialised; it is
/// re-initialised as an empty [`OnceLock`] on deserialisation, and the MPC
/// table will be fetched from the network again on the first access.
#[derive(serde::Serialize, serde::Deserialize)]
struct ObserverDatasetSerde {
    custom_observers: Vec<Observer>,
    mpc_error_model: Option<ObsErrorModel>,
}

impl serde::Serialize for ObserverDataset {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ObserverDatasetSerde {
            custom_observers: self.custom_observers.clone(),
            mpc_error_model: self.mpc_error_model,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ObserverDataset {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let proxy = ObserverDatasetSerde::deserialize(deserializer)?;
        Ok(ObserverDataset::new(
            proxy.custom_observers,
            proxy.mpc_error_model,
        ))
    }
}
