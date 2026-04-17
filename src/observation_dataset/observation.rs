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
    astrometry::EquCoord,
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
    /// Zero-based index of this observation in the source dataset, assigned during construction.
    pub(crate) index: ObsIndex,

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
    /// Return the zero-based position of this observation in its parent dataset's storage vector.
    ///
    /// # Returns
    ///
    /// The `ObsIndex` (a `usize`) assigned to this observation during dataset construction.
    pub fn index(&self) -> ObsIndex {
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
