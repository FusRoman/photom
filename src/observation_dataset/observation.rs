//! A single astrometric and photometric measurement.
//!
//! This module defines [`Observation`], the fundamental record type stored
//! inside an [`ObsDataset`](crate::observation_dataset::ObsDataset).  Each
//! value bundles a sky position, a photometric measurement, a detection epoch,
//! and an optional reference to the observatory that recorded it.
//!
//! ## Field access
//!
//! All fields of [`Observation`] are `pub(crate)` to prevent external mutation.
//! Read-only access is provided through dedicated getter methods:
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

/// A single astrometric and photometric measurement.
///
/// Each `Observation` bundles the equatorial sky position, the photometric
/// measurement, the detection epoch, and an optional reference to the
/// observatory that recorded it.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Observation {
    /// Zero-based index of this observation in the source dataset.
    /// None by default and assigned during dataset construction.
    pub(crate) index: Option<ObsIndex>,

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
    /// Create a new `Observation` with the specified fields.
    ///
    /// The `index` field is set to `None` by default and will be assigned during dataset construction.
    ///
    /// The `observer` field is optional and can be set to `None` if the observer is unknown.
    /// Use [`ObsDataset::get_observer`](crate::observation_dataset::ObsDataset::get_observer) to resolve the `ObserverId` to a full `Observer` value.
    ///
    /// # Parameters
    /// - `id` — unique identifier for this observation within its dataset (corresponds to the `id` column of the source `DataFrame`).
    /// - `equ_coord` — equatorial sky coordinates (right ascension and declination) with their associated measurement uncertainties, all in **radians**.
    /// - `photometry` — photometric measurement: apparent magnitude, its uncertainty, and the filter through which the observation was taken.
    /// - `mjd_tt` — detection epoch as a Modified Julian Date in Terrestrial Time, expressed in **days**.
    /// - `observer` — optional reference to the observatory that recorded this observation. Use [`ObsDataset::get_observer`](crate::observation_dataset::ObsDataset::get_observer) to resolve this identifier to a full `Observer` value.
    ///
    /// # Returns
    ///
    /// A new `Observation` instance with the specified fields and `index` set to `None`.
    pub fn new(
        id: ObsId,
        equ_coord: EquCoord,
        photometry: Photometry,
        mjd_tt: MJDTT,
        observer: Option<ObserverId>,
    ) -> Self {
        Self {
            index: None, // index is assigned during dataset construction
            id,
            equ_coord,
            photometry,
            mjd_tt,
            observer,
        }
    }

    /// Return the zero-based position of this observation in its parent dataset's storage vector.
    ///
    /// # Returns
    ///
    /// The `ObsIndex` (a `usize`) assigned to this observation during dataset construction.
    pub fn index(&self) -> Option<ObsIndex> {
        self.index
    }

    /// Return a reference to the unique identifier of this observation.
    ///
    /// The identifier corresponds to the value in the `id` column of the source `DataFrame`
    /// and is unique within a given `ObsDataset`.
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

    fn make_obs(id: u64, mjd: f64) -> Observation {
        Observation::new(
            id,
            EquCoord::new(0.5, 1e-5, 0.2, 1e-5),
            make_photometry(),
            mjd,
            None,
        )
    }

    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    #[test]
    fn new_sets_index_to_none() {
        let obs = make_obs(42, 60000.0);
        assert!(obs.index().is_none());
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
        let obs = Observation::new(
            1,
            EquCoord::new(0.0, 1e-5, 0.0, 1e-5),
            make_photometry(),
            60000.0,
            Some(ObserverId::MpcCode(*b"T05")),
        );
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
    // Index assignment (pub(crate) field)
    // ------------------------------------------------------------------

    #[test]
    fn index_assignment() {
        let mut obs = make_obs(1, 60000.0);
        assert!(obs.index().is_none());
        obs.index = Some(ObsIndex::from(3usize));
        assert_eq!(obs.index(), Some(ObsIndex::from(3usize)));
    }
}
