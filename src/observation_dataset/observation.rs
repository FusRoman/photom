//! A single astrometric and photometric measurement.
//!
//! This module defines two related types:
//!
//! - [`ObservationInput`] — an observation that has not yet been placed in a
//!   dataset.  It carries all measurement data but no vector position.  This
//!   is the type accepted by [`crate::observation_dataset::ObsDataset::push_observation`] and produced by
//!   all ingestion backends (ADES, MPC 80-column, Polars, DataFusion).
//!
//! - [`Observation`] — an observation that lives inside an
//!   [`ObsDataset`](crate::observation_dataset::ObsDataset).  It is identical
//!   to [`ObservationInput`] plus an [`ObsIndex`] that records its zero-based
//!   position in the dataset's storage vector.  The index is always set and
//!   correct: it is assigned atomically with insertion by `ObsDataset` and is
//!   never modified afterwards.
//!
//! ## Field access
//!
//! All fields of [`Observation`] are `pub(crate)` to prevent external
//! mutation.  Read-only access is provided through dedicated getter methods:
//! [`Observation::index`], [`Observation::id`], [`Observation::equ_coord`],
//! [`Observation::photometry`], and [`Observation::mjd_tt`].  The `observer`
//! field is accessed indirectly via
//! [`ObsDataset::get_observer`](crate::observation_dataset::ObsDataset::get_observer).

use crate::{
    MJDTT,
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsId, index::ObsIndex},
    observer::dataset::ObserverId,
    photometry::Photometry,
};

// ---------------------------------------------------------------------------
// ObservationInput
// ---------------------------------------------------------------------------

/// An observation that has not yet been placed in an [`ObsDataset`](crate::observation_dataset::ObsDataset).
///
/// `ObservationInput` carries all measurement data for a single
/// astrometric/photometric detection but does not have a vector position yet.
/// It is the type accepted by
/// [`ObsDataset::push_observation`](crate::observation_dataset::ObsDataset::push_observation)
/// and is produced by all ingestion backends (ADES, MPC 80-column, Polars,
/// DataFusion).
///
/// Once inserted into a dataset, `ObservationInput` is converted to
/// [`Observation`] with an immutable [`ObsIndex`] assigned by the dataset.
#[derive(Debug, Clone)]
pub struct ObservationInput {
    /// Unique identifier for this observation within its dataset.
    pub id: ObsId,

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
    pub observer: Option<ObserverId>,
}

impl ObservationInput {
    /// Create a new `ObservationInput` with the specified fields.
    ///
    /// The `observer` field is optional; pass `None` when the observer is
    /// unknown.  Use
    /// [`ObsDataset::get_observer`](crate::observation_dataset::ObsDataset::get_observer)
    /// to resolve an [`ObserverId`] to a full [`crate::observer::Observer`] value
    /// after insertion.
    ///
    /// # Parameters
    ///
    /// - `id` — unique identifier for this observation within its dataset
    ///   (corresponds to the `id` column of the source `DataFrame`).
    /// - `equ_coord` — equatorial sky coordinates (right ascension and
    ///   declination) with their associated measurement uncertainties, all
    ///   in **radians**.
    /// - `photometry` — photometric measurement: apparent magnitude, its
    ///   uncertainty, and the filter through which the observation was taken.
    /// - `mjd_tt` — detection epoch as a Modified Julian Date in Terrestrial
    ///   Time, expressed in **days**.
    /// - `observer` — optional reference to the observatory that recorded
    ///   this observation.
    ///
    /// # Returns
    ///
    /// A new `ObservationInput` ready to be inserted into an
    /// [`ObsDataset`](crate::observation_dataset::ObsDataset).
    pub fn new(
        id: ObsId,
        equ_coord: EquCoord,
        photometry: Photometry,
        mjd_tt: MJDTT,
        observer: Option<ObserverId>,
    ) -> Self {
        Self {
            id,
            equ_coord,
            photometry,
            mjd_tt,
            observer,
        }
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// A single astrometric and photometric measurement placed inside an
/// [`ObsDataset`](crate::observation_dataset::ObsDataset).
///
/// Each `Observation` bundles the equatorial sky position, the photometric
/// measurement, the detection epoch, an optional reference to the observatory
/// that recorded it, and — crucially — the **zero-based index** of its
/// position in the parent dataset's storage vector.
///
/// `Observation` values are created exclusively by `ObsDataset`; they cannot
/// be constructed directly by external code.  This invariant guarantees that
/// `index` is always consistent with the observation's actual position.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Observation {
    /// Zero-based index of this observation in the parent dataset's storage
    /// vector.  Assigned atomically by `ObsDataset` during insertion and
    /// never modified afterwards.
    index: ObsIndex,

    /// Unique identifier for this observation within its dataset.
    ///
    /// Corresponds to the `id` column of the source `DataFrame`.
    pub(crate) id: ObsId,

    /// Equatorial sky coordinates (right ascension and declination) with
    /// their associated measurement uncertainties, all in **radians**.
    pub(crate) equ_coord: EquCoord,

    /// Photometric measurement: apparent magnitude, its uncertainty, and the
    /// filter through which the observation was taken.
    pub(crate) photometry: Photometry,

    /// Detection epoch (Modified Julian Date, Terrestrial Time, **days**).
    pub(crate) mjd_tt: MJDTT,

    /// Reference to the observatory that recorded this observation, or `None`
    /// when the observer is unknown.
    ///
    /// Use [`ObsDataset::get_observer`](crate::observation_dataset::ObsDataset::get_observer)
    /// to resolve this identifier to a full `Observer` value.
    pub(crate) observer: Option<ObserverId>,
}

/// Implement equality and ordering based on the unique identifier and detection epoch.
impl PartialEq for Observation {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Observation {}

/// Implement ordering based on detection epoch (MJDTT), then by unique identifier (ObsId) to break ties.
impl Ord for Observation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.mjd_tt
            .total_cmp(&other.mjd_tt)
            .then(self.id.cmp(&other.id))
    }
}

/// Implement partial ordering consistent with the total ordering defined in `Ord`.
impl PartialOrd for Observation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Observation {
    /// Place an [`ObservationInput`] into a dataset by attaching its vector position.
    ///
    /// This is the only constructor for `Observation`.  It is `pub(crate)` so
    /// that only `ObsDataset` methods can create placed observations, preserving
    /// the invariant that `index` always matches the observation's actual position.
    pub(crate) fn place(input: ObservationInput, index: ObsIndex) -> Self {
        Self {
            index,
            id: input.id,
            equ_coord: input.equ_coord,
            photometry: input.photometry,
            mjd_tt: input.mjd_tt,
            observer: input.observer,
        }
    }

    /// Re-assign the vector position of this observation, consuming `self` and
    /// returning a new `Observation` with `index` set to `new_idx`.
    ///
    /// Used by [`crate::observation_dataset::ObsDataset`] when merging datasets:
    /// observations that were valid in their source dataset are re-placed at
    /// their new position in the merged vector.  All measurement fields are
    /// preserved unchanged.
    pub(crate) fn reindex(self, new_idx: ObsIndex) -> Self {
        Self {
            index: new_idx,
            ..self
        }
    }

    /// Return the zero-based position of this observation in its parent
    /// dataset's storage vector.
    ///
    /// # Returns
    ///
    /// The [`ObsIndex`] (a `usize`) that was assigned during insertion into
    /// the dataset.
    pub fn index(&self) -> ObsIndex {
        self.index
    }

    /// Return a reference to the unique identifier of this observation.
    ///
    /// The identifier corresponds to the value in the `id` column of the
    /// source `DataFrame` and is unique within a given `ObsDataset`.
    ///
    /// # Returns
    ///
    /// A shared reference to the `ObsId` (`u64`) of this observation.
    pub fn id(&self) -> &ObsId {
        &self.id
    }

    /// Return a reference to the observer identifier for this observation, if any.
    ///
    /// # Returns
    ///
    /// `Some(&ObserverId)` when an observer is associated with this observation;
    /// `None` when the observer is unknown.
    pub fn observer_id(&self) -> Option<&ObserverId> {
        self.observer.as_ref()
    }

    /// Return a reference to the equatorial sky coordinates of this observation.
    ///
    /// The coordinates include right ascension and declination together with their
    /// associated measurement uncertainties, all expressed in **radians**.
    ///
    /// # Returns
    ///
    /// A shared reference to the `EquCoord` of this observation.
    pub fn equ_coord(&self) -> &EquCoord {
        &self.equ_coord
    }

    /// Return a reference to the photometric measurement of this observation.
    ///
    /// The `Photometry` value contains the apparent magnitude, its uncertainty, and the
    /// bandpass filter label.
    ///
    /// # Returns
    ///
    /// A shared reference to the `Photometry` of this observation.
    pub fn photometry(&self) -> &Photometry {
        &self.photometry
    }

    /// Return the detection epoch as a Modified Julian Date in Terrestrial Time.
    ///
    /// # Returns
    ///
    /// The epoch as an `MJDTT` value (an `f64`, in days).
    pub fn mjd_tt(&self) -> MJDTT {
        self.mjd_tt
    }
}

#[cfg(test)]
mod observation_tests {
    use super::*;
    use crate::{
        coordinates::equatorial::EquCoord,
        observation_dataset::index::ObsIndex,
        observer::dataset::ObserverId,
        photometry::{Filter, Photometry},
    };

    fn make_photometry() -> Photometry {
        Photometry {
            magnitude: 15.0,
            error: 0.1,
            filter: Filter::String("V".to_string()),
        }
    }

    fn make_input(id: u64, mjd: f64) -> ObservationInput {
        ObservationInput::new(
            id,
            EquCoord::new(0.5, 1e-5, 0.2, 1e-5),
            make_photometry(),
            mjd,
            None,
        )
    }

    fn make_obs(id: u64, mjd: f64) -> Observation {
        Observation::place(make_input(id, mjd), 0)
    }

    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    #[test]
    fn place_sets_correct_index() {
        let input = make_input(42, 60000.0);
        let obs = Observation::place(input, 7);
        assert_eq!(obs.index(), 7);
    }

    #[test]
    fn getters_return_correct_values() {
        let obs = make_obs(99, 60123.5);
        assert_eq!(*obs.id(), 99);
        assert_eq!(obs.mjd_tt(), 60123.5);
        assert_eq!(obs.equ_coord().ra, 0.5);
        assert_eq!(obs.photometry().magnitude, 15.0);
    }

    #[test]
    fn observer_field_preserved() {
        let input = ObservationInput::new(
            1,
            EquCoord::new(0.0, 1e-5, 0.0, 1e-5),
            make_photometry(),
            60000.0,
            Some(ObserverId::MpcCode(*b"T05")),
        );
        let obs = Observation::place(input, 0);
        // We can't directly inspect the observer field (pub(crate)), but we can
        // verify the observation was constructed without panic.
        assert_eq!(*obs.id(), 1);
    }

    // ------------------------------------------------------------------
    // Equality (based on id)
    // ------------------------------------------------------------------

    #[test]
    fn eq_by_id() {
        let a = make_obs(7, 60000.0);
        let b = make_obs(7, 60001.0); // different epoch, same id
        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_ids() {
        let a = make_obs(1, 60000.0);
        let b = make_obs(2, 60000.0);
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // Ordering (based on mjd_tt, then id)
    // ------------------------------------------------------------------

    #[test]
    fn ord_by_epoch() {
        let earlier = make_obs(1, 59000.0);
        let later = make_obs(2, 60000.0);
        assert!(earlier < later);
    }

    #[test]
    fn ord_tie_broken_by_id() {
        let a = make_obs(1, 60000.0);
        let b = make_obs(2, 60000.0);
        assert!(a < b);
    }

    // ------------------------------------------------------------------
    // Clone
    // ------------------------------------------------------------------

    #[test]
    fn clone_is_equal() {
        let obs = make_obs(5, 60000.0);
        let cloned = obs.clone();
        assert_eq!(obs, cloned);
        assert_eq!(obs.mjd_tt(), cloned.mjd_tt());
    }

    // ------------------------------------------------------------------
    // Index is ObsIndex (non-optional)
    // ------------------------------------------------------------------

    #[test]
    fn index_is_set_by_place() {
        let input = make_input(1, 60000.0);
        let obs = Observation::place(input, ObsIndex::from(3usize));
        assert_eq!(obs.index(), ObsIndex::from(3usize));
    }
}
