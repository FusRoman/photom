//! Core observation data types for the photom crate.
//!
//! This module defines the fundamental building blocks used throughout the
//! pipeline: individual astrometric/photometric measurements ([`Observation`]),
//! the dataset that holds a collection of them ([`ObsDataset`]), the
//! identifier types that label observations, nights, and observatories
//! ([`ObsId`], [`NightId`], [`ObserverId`]), and the error type that covers
//! all failure modes arising during dataset construction ([`ObsDatasetError`]).
//!
//! ## Key design notes
//!
//! - **LRU cache** — [`ObsDataset`] keeps a least-recently-used cache of up
//!   to 1 000 [`Observation`] values so that repeated look-ups by [`ObsId`]
//!   do not scan the full observation list on every call.
//! - **Lazy MPC initialisation** — the Minor Planet Center observatory table
//!   is fetched from the network only on the first call to
//!   [`ObsDataset::get_observer`] for an MPC-coded site, and the result
//!   (success *or* failure) is stored in a [`std::sync::OnceLock`] so that
//!   subsequent calls are free.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`ObsId`] | type alias | Unique numeric identifier for a single observation |
//! | [`NightId`] | struct | Logical identifier for a night of observation |
//! | [`ObserverId`] | enum | Reference to either a custom or an MPC-coded observer |
//! | [`Observation`] | struct | A single astrometric/photometric measurement |
//! | [`ObsDataset`] | struct | Collection of observations with lazy observer resolution |
//! | [`ObsDatasetError`] | enum | Errors arising from dataset construction |

use std::{num::NonZeroUsize, sync::OnceLock};

use lru::LruCache;
use polars::frame::DataFrame;
use std::time::Duration;
use thiserror::Error;
use ureq::Agent;

use crate::{
    MJDTT,
    astrometry::EquCoord,
    io::polars::{error::PolarsError, load_observation_from_polars},
    observer::{
        Observer,
        error_model::{ErrorModelParseError, ObsErrorModel},
        mpc::{MPCError, MpcCode, MpcCodeObs, init_observatories},
    },
    photometry::Photometry,
};

/// Unique numeric identifier for a single observation.
///
/// Observations are keyed by this value inside [`ObsDataset`] and its
/// internal LRU cache.  The identifier is assigned by the data source (e.g.
/// the `id` column of a Polars [`DataFrame`]) and must be unique within a
/// dataset.
pub type ObsId = u64;

/// Logical identifier for a night of observation.
///
/// Wraps a `u32` that typically represents an integer MJD day number
/// (e.g. `60312`).  The value must be stable across runs because it is used
/// as a directory name in on-disk outputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NightId(pub u32);

/// Reference to the observer associated with an [`Observation`].
///
/// An observer can be identified in one of two ways:
///
/// - **[`ObserverId::IntId`]** — an index into the `custom_observers` list
///   stored inside the parent [`ObsDataset`].  Used for geodetic sites
///   supplied directly in the input data.
/// - **[`ObserverId::MpcCode`]** — a three-byte ASCII Minor Planet Center
///   observatory code (e.g. `b"I41"`).  The corresponding [`Observer`]
///   metadata is resolved lazily from the MPC catalogue on the first access.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObserverId {
    /// Index into the dataset's internal list of custom geodetic observers.
    IntId(usize),
    /// Three-byte ASCII MPC observatory code (e.g. `b"G96"`).
    MpcCode(MpcCode),
}

/// A single astrometric and photometric measurement.
///
/// Each `Observation` bundles the equatorial sky position, the photometric
/// measurement, the detection epoch, and an optional reference to the
/// observatory that recorded it.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Unique identifier for this observation within its dataset.
    ///
    /// Corresponds to the `id` column of the source [`DataFrame`].
    pub id: ObsId,

    /// Night during which the observation was recorded, if known.
    ///
    /// `None` when the night assignment has not yet been computed or is not
    /// available in the source data.
    pub night_id: Option<NightId>,

    /// Equatorial sky coordinates (right ascension and declination) with
    /// their associated measurement uncertainties, all in **radians**.
    pub equ_coord: EquCoord,

    /// Photometric measurement: apparent magnitude, its uncertainty, and the
    /// filter through which the observation was taken.
    pub photometry: Photometry,

    /// Detection epoch (Modified Julian Date, Terrestrial Time, **days**).
    pub mjd_tt: MJDTT,

    /// Reference to the observatory that recorded this observation, or `None`
    /// when the observer is unknown.
    ///
    /// Use [`ObsDataset::get_observer`] to resolve this identifier to a full
    /// [`Observer`] value.
    pub observer: Option<ObserverId>,
}

/// Errors that can arise when constructing or using an [`ObsDataset`].
#[derive(Debug, Error)]
pub enum ObsDatasetError {
    /// The network request to the Minor Planet Center catalogue failed.
    #[error(transparent)]
    MPCError(#[from] MPCError),

    /// The astrometric error-model file could not be parsed.
    #[error(transparent)]
    ErrorModelError(#[from] ErrorModelParseError),

    /// A Polars I/O or schema error occurred while loading observations.
    #[error(transparent)]
    PolarIoError(#[from] PolarsError),
}

/// A collection of [`Observation`]s with associated observer metadata.
///
/// `ObsDataset` is the primary container for observation data in the pipeline.
/// In addition to the raw observations it holds:
///
/// - A list of **custom geodetic observers** supplied directly in the input,
///   referenced by index through [`ObserverId::IntId`].
/// - A **lazily-initialised MPC lookup table** that maps three-byte MPC codes
///   to [`Observer`] metadata.  The table is fetched from the MPC website
///   on the first access and cached for the lifetime of the dataset.
/// - An **LRU cache** of up to 1 000 [`Observation`] values so that repeated
///   look-ups by [`ObsId`] avoid a full linear scan.
pub struct ObsDataset {
    /// Full list of observations in insertion order.
    observations: Vec<Observation>,

    /// Geodetic observers supplied by the input data, stored once and
    /// referenced by index to avoid duplication.
    custom_observers: Vec<Observer>,

    /// Lazily-initialised MPC observatory lookup table.
    ///
    /// Populated on the first call to [`ObsDataset::mpc_observers`].  If
    /// initialisation fails the error is stored here and re-returned on every
    /// subsequent call without retrying the network request.
    mpc_observers: OnceLock<Result<MpcCodeObs, ObsDatasetError>>,

    /// Astrometric error model used to assign measurement accuracies to
    /// MPC-coded observers during MPC table initialisation.
    mpc_error_model: ObsErrorModel,

    /// LRU cache keyed by [`ObsId`] with a fixed capacity of 1 000 entries.
    ///
    /// Entries are cloned into the cache on first access and evicted in
    /// least-recently-used order when the cache is full.
    lru_cache_obs: LruCache<ObsId, Observation>,
}

impl ObsDataset {
    /// Construct an [`ObsDataset`] from a Polars [`DataFrame`].
    ///
    /// Validates the frame against the expected schema, extracts all
    /// observation columns, and assembles the dataset.  See
    /// [`crate::io::polars`] for the full column specification and
    /// observer-resolution rules.
    /// 
    /// # Arguments
    /// 
    /// - `df` — the source Polars [`DataFrame`] containing the observation data.
    /// - `error_model` — the [`ObsErrorModel`] used to assign astrometric accuracies to 
    ///     MPC-coded observers during MPC table initialisation.
    /// - `lru_cache_size` — optional capacity for the LRU cache used to speed up repeated observation lookups; 
    ///     if `None`, the cache size is set to 1 000.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if the frame fails schema validation, if a
    /// Polars-internal operation fails, or if any observer column violates
    /// the resolution rules (e.g. a partially-null geodetic triplet).
    pub fn from_polars(df: &DataFrame, error_model: ObsErrorModel, lru_cache_size: Option<usize>) -> Result<Self, PolarsError> {
        load_observation_from_polars(df, error_model, lru_cache_size)
    }

    /// Look up a single observation by its [`ObsId`].
    ///
    /// Returns a shared reference to the matching [`Observation`], or `None`
    /// if no observation with the given `id` exists in this dataset.
    ///
    /// ## Caching strategy
    ///
    /// To avoid repeatedly scanning the full observation list, results are
    /// stored in an LRU cache (capacity 1 000).  The look-up proceeds in two
    /// phases:
    ///
    /// 1. **Cache probe** — [`LruCache::contains`] is called first.  If the
    ///    entry is present, [`LruCache::get`] is called in a separate
    ///    statement to obtain the reference.  This two-step approach is
    ///    necessary because a single `get` call borrows `self` mutably (to
    ///    update the LRU order) and would prevent returning a reference into
    ///    the same `self`; the intermediate `contains` check lets the
    ///    compiler prove the borrows do not overlap.
    /// 2. **Linear scan** — if the cache misses, the `observations` list is
    ///    searched with [`Iterator::find`].  The found value is cloned into
    ///    the cache before a reference is returned, so subsequent look-ups
    ///    for the same `id` hit the cache.
    pub fn get_observation(&mut self, id: ObsId) -> Option<&Observation> {
        if self.lru_cache_obs.contains(&id) {
            return self.lru_cache_obs.get(&id);
        }
        let obs = self.observations.iter().find(|obs| obs.id == id)?.clone();
        self.lru_cache_obs.put(id, obs);
        self.lru_cache_obs.get(&id)
    }

    /// Return an iterator over all observations in insertion order.
    ///
    /// The iterator yields shared references and does not clone any data.
    /// The order matches the order of the source [`DataFrame`] rows.
    pub fn iter_observations(&self) -> impl Iterator<Item = &Observation> {
        self.observations.iter()
    }

    /// Look up the [`Observer`] associated with a given observation.
    ///
    /// Returns `None` if the observation does not exist, if it has no
    /// observer, or if the MPC catalogue could not be initialised.
    ///
    /// ## Borrow-checker note
    ///
    /// [`ObserverId`] is `Copy`, so the observer identifier is copied out of
    /// the [`Observation`] returned by [`ObsDataset::get_observation`] in a
    /// single statement.  This releases the mutable borrow on `self` held by
    /// `get_observation` before `custom_observers` or `mpc_observers` are
    /// accessed, satisfying the borrow checker without any heap allocation.
    pub fn get_observer(&mut self, id: ObsId) -> Option<&Observer> {
        // Copy the ObserverId out first to release the borrow on `self` held by
        // `get_observation` before we access `self.custom_observers` or
        // `self.mpc_observers()`.  ObserverId is Copy so no allocation occurs.
        let observer_id = self.get_observation(id)?.observer?;
        match observer_id {
            ObserverId::IntId(idx) => self.custom_observers.get(idx),
            ObserverId::MpcCode(code) => self.mpc_observers().ok()?.get(&code),
        }
    }

    /// Create a new dataset from pre-parsed data.
    ///
    /// This constructor is used internally by [`ObsDataset::from_polars`] and
    /// by test helpers.  The LRU cache is initialised with a fixed capacity of
    /// **1 000** entries; the MPC observatory table is not fetched until the
    /// first call to [`ObsDataset::get_observer`] for an MPC-coded site.
    ///
    /// # Arguments
    ///
    /// - `observations`     — the full list of observations.
    /// - `custom_observers` — geodetic observers de-duplicated by the caller.
    /// - `error_model`      — astrometric error model used during MPC
    ///   observatory initialisation.
    /// - `lru_cache_size`    — optional capacity for the LRU cache (default 1 000).
    pub(crate) fn new(
        observations: Vec<Observation>,
        custom_observers: Vec<Observer>,
        error_model: ObsErrorModel,
        lru_cache_size: Option<usize>,
    ) -> Self {
        Self {
            observations,
            custom_observers,
            mpc_observers: OnceLock::new(),
            mpc_error_model: error_model,
            lru_cache_obs: LruCache::new(
                NonZeroUsize::new(lru_cache_size.unwrap_or(1000)).unwrap(),
            ),
        }
    }

    /// Returns a reference to the MPC observatory lookup table, initializing
    /// it on the first call by fetching data from the MPC website.
    ///
    /// The result is cached: subsequent calls return immediately without any
    /// I/O. If initialization failed, the error is returned on every call.
    ///
    /// # Errors
    ///
    /// Returns [`ObsDatasetError::ErrorModelError`] if the error model file
    /// cannot be parsed, or [`ObsDatasetError::MPCError`] if the MPC fetch
    /// fails.
    pub(crate) fn mpc_observers(&self) -> Result<&MpcCodeObs, &ObsDatasetError> {
        self.mpc_observers
            .get_or_init(|| {
                let config = Agent::config_builder()
                    .timeout_global(Some(Duration::from_secs(10)))
                    .build();
                let agent: Agent = config.into();

                let error_model_data = self.mpc_error_model.read_error_model_file()?;
                let obs = init_observatories(agent, &error_model_data)?;
                Ok(obs)
            })
            .as_ref()
    }
}
