//! Observer dataset: custom geodetic observers and lazy MPC catalogue resolution.
//!
//! This module provides `ObserverDataset` (crate-private), the internal
//! container that aggregates two sources of observatory metadata:
//!
//! - **Custom geodetic observers** — supplied directly in the input data and
//!   stored in an indexed `Vec<Observer>`.  Each unique geodetic site is
//!   interned once; observations reference it by its `Vec` index via
//!   [`ObserverId::IntId`].
//!
//! - **MPC-coded observers** — identified by a three-byte ASCII Minor Planet
//!   Center code (e.g. `b"G96"`).  The full lookup table is fetched from the
//!   MPC website on the first access and cached in a [`std::sync::OnceLock`],
//!   so subsequent calls incur no I/O overhead.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`ObserverId`] | enum | Reference to a custom or MPC-coded observer |
//!
//! The `ObserverDataset` struct itself is `pub(crate)` and is not part of
//! the public API.

use std::{
    fmt::{self, Display},
    sync::OnceLock,
};

use crate::{
    observation_dataset::ObsDatasetError,
    observer::{
        Observer,
        error_model::ObsErrorModel,
        mpc::{MpcCode, MpcCodeObs, init_observatories},
    },
};

/// Reference to the observer associated with an observation.
///
/// An observer can be identified in one of two ways:
///
/// - **[`ObserverId::IntId`]** — an index into the `custom_observers` list
///   stored inside the parent observation dataset.  Used for geodetic sites
///   supplied directly in the input data.
/// - **[`ObserverId::MpcCode`]** — a three-byte ASCII Minor Planet Center
///   observatory code (e.g. `b"I41"`).  The corresponding [`Observer`]
///   metadata is resolved lazily from the MPC catalogue on the first access.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ObserverId {
    /// Index into the dataset's internal list of custom geodetic observers.
    IntId(usize),
    /// Three-byte ASCII MPC observatory code (e.g. `b"G96"`).
    MpcCode(MpcCode),
}

impl Display for ObserverId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObserverId::IntId(id) => write!(f, "Observer#{id}"),
            ObserverId::MpcCode(code) => {
                let s = std::str::from_utf8(code).unwrap_or("???");
                write!(f, "MPC({s})")
            }
        }
    }
}

/// Internal container for all observer metadata used by an observation dataset.
///
/// [`ObserverDataset`] unifies two independent sources of observatory
/// information behind a single [`get`](ObserverDataset::get) call:
///
/// - **Custom observers** stored at construction time in a `Vec`, addressed by
///   [`ObserverId::IntId`] index.
/// - **MPC-coded observers** fetched lazily from the Minor Planet Center on the
///   first [`ObserverId::MpcCode`] lookup and cached thereafter.
///
/// The lazy fetch is protected by a [`OnceLock`]: once the result — whether
/// success or failure — is stored, no further network I/O is performed.
#[derive(Debug)]
pub struct ObserverDataset {
    /// Geodetic observers supplied by the input data, stored once and
    /// referenced by index to avoid duplication.
    pub(crate) custom_observers: Vec<Observer>,

    /// Lazily-initialised MPC observatory lookup table.
    ///
    /// Populated on the first successful call to [`ObserverDataset::mpc_observers`].
    /// If initialisation fails, the lock remains uninitialised and the next
    /// call will retry.
    mpc_observers: OnceLock<MpcCodeObs>,

    /// Astrometric error model used to assign measurement accuracies to
    /// MPC-coded observers during MPC table initialisation.
    pub(crate) mpc_error_model: Option<ObsErrorModel>,
}

impl Clone for ObserverDataset {
    fn clone(&self) -> Self {
        Self {
            custom_observers: self.custom_observers.clone(),
            mpc_observers: OnceLock::new(), // reset: will re-fetch on next call
            mpc_error_model: self.mpc_error_model,
        }
    }
}

impl ObserverDataset {
    /// Create a new [`ObserverDataset`] with a pre-built list of custom observers
    /// and an astrometric error model for MPC site lookup.
    ///
    /// The MPC observatory table is **not** fetched at this point; it is
    /// initialised lazily on the first call to
    /// [`mpc_observers`](ObserverDataset::mpc_observers).
    ///
    /// # Arguments
    ///
    /// - `custom_observers` — pre-built list of geodetic [`Observer`] values,
    ///   addressable by their position index via [`ObserverId::IntId`].
    /// - `mpc_error_model` — the astrometric error model variant used to
    ///   populate per-site measurement accuracies when the MPC table is loaded.
    ///
    /// # Returns
    ///
    /// A freshly constructed [`ObserverDataset`] with an uninitialised MPC
    /// lookup table.
    #[cfg_attr(not(feature = "polars"), allow(dead_code))]
    pub(crate) fn new(
        custom_observers: Vec<Observer>,
        mpc_error_model: Option<ObsErrorModel>,
    ) -> Self {
        Self {
            custom_observers,
            mpc_observers: OnceLock::new(),
            mpc_error_model,
        }
    }

    /// Create an empty [`ObserverDataset`] with no custom observers and an optional
    /// astrometric error model for MPC site lookup.
    ///
    /// The MPC observatory table is **not** fetched at this point; it is
    /// initialised lazily on the first call to [`get`](ObserverDataset::get).
    ///
    /// # Arguments
    ///
    /// - `error_model` — the astrometric error model variant used to populate
    ///   per-site measurement accuracies when the MPC table is loaded.
    ///
    /// # Returns
    ///
    /// A freshly constructed [`ObserverDataset`] with an uninitialised MPC
    /// lookup table and an empty custom observer list.
    pub fn empty(error_model: Option<ObsErrorModel>) -> Self {
        Self {
            custom_observers: Vec::new(),
            mpc_observers: OnceLock::new(),
            mpc_error_model: error_model,
        }
    }

    /// Look up the [`Observer`] associated with the given [`ObserverId`].
    ///
    /// - For [`ObserverId::IntId`] the custom observer list is indexed
    ///   directly (O(1)).
    /// - For [`ObserverId::MpcCode`] the MPC lookup table is queried, being
    ///   fetched from the network if this is the first call.
    ///
    /// # Arguments
    ///
    /// - `obs_id` — reference to the observer identifier to resolve.
    ///
    /// # Returns
    ///
    /// `Some(&Observer)` if the identifier is valid and the corresponding
    /// observer is found, or `None` if the index is out of range, the MPC
    /// code is not present in the catalogue, or the MPC table failed to load.
    pub fn get(&self, obs_id: &ObserverId) -> Option<&Observer> {
        match obs_id {
            ObserverId::IntId(idx) => self.custom_observers.get(*idx),
            ObserverId::MpcCode(code) => self.mpc_observers().ok()?.get(code),
        }
    }

    /// Merge another [`ObserverDataset`] into this one.
    ///
    /// Custom observers from `other` are appended to `self.custom_observers`.
    /// Returns the offset that was applied (i.e. the original length of
    /// `self.custom_observers` before the merge), so the caller can shift any
    /// `ObserverId::IntId` values from `other` accordingly.
    ///
    /// The MPC observers and error model from `other` are discarded; `self`
    /// retains its own MPC cache and error model.
    #[cfg_attr(not(any(feature = "ades", feature = "mpc_80_col")), allow(dead_code))]
    pub(crate) fn merge_custom_observers(&mut self, other: ObserverDataset) -> usize {
        let offset = self.custom_observers.len();
        self.custom_observers.extend(other.custom_observers);
        offset
    }

    /// Returns a reference to the MPC observatory lookup table, initialising
    /// it on demand if not yet available.
    ///
    /// Initialisation fetches the MPC observatory list and applies the
    /// astrometric error model supplied via
    /// [`ObserverDataset::mpc_error_model`].  On success the result is cached
    /// in a [`OnceLock`]: subsequent calls return immediately without any I/O.
    ///
    /// If initialisation fails the lock is left uninitialised, so the next
    /// call will retry — unlike a pattern that caches the error permanently.
    ///
    /// # Concurrency
    ///
    /// Safe to call from multiple threads.  If two threads race during the
    /// first initialisation, both perform the work independently; the winner
    /// of `OnceLock::set` stores its result and the loser's value is silently
    /// discarded.  All threads ultimately observe the same cached table.
    ///
    /// # Returns
    ///
    /// A shared reference to the [`MpcCodeObs`] map on success.
    ///
    /// # Errors
    ///
    /// - [`ObsDatasetError::ErrorModelNotFound`] — no error model has been
    ///   attached via [`ObsDataset::with_error_model`].
    /// - [`ObsDatasetError::MPCError`] — the MPC fetch or table parsing failed.
    pub(crate) fn mpc_observers(&self) -> Result<&MpcCodeObs, ObsDatasetError> {
        if let Some(obs) = self.mpc_observers.get() {
            return Ok(obs);
        }

        // Not yet initialised (or previous attempt failed) — try now.
        let error_model_data = self
            .mpc_error_model
            .as_ref()
            .ok_or(ObsDatasetError::ErrorModelNotFound)?
            .read_error_model_file()?;

        let obs = init_observatories(&error_model_data)?;

        // Another thread may have initialised concurrently — `set` is a no-op
        // if the lock is already filled, and `get` will return the winner.
        let _ = self.mpc_observers.set(obs);

        Ok(self
            .mpc_observers
            .get()
            .expect("just set above or by a concurrent thread"))
    }
}
