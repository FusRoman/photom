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
    /// Use [`ObsDataset::get_observer`] to resolve this identifier to a full
    /// [`Observer`] value.
    pub(crate) observer: Option<ObserverId>,
}

impl Observation {
    pub fn index(&self) -> ObsIndex {
        self.index
    }

    pub fn id(&self) -> &ObsId {
        &self.id
    }

    pub fn equ_coord(&self) -> &EquCoord {
        &self.equ_coord
    }

    pub fn photometry(&self) -> &Photometry {
        &self.photometry
    }

    pub fn mjd_tt(&self) -> MJDTT {
        self.mjd_tt
    }
}
