#![cfg(feature = "ades")]
use camino::Utf8Path;

use crate::{Arcseconds, io::ades::parse_ades_file, observation_dataset::ObsDataset};

impl ObsDataset {
    /// Build an [`ObsDataset`] by reading an ADES XML file.
    ///
    /// Both the *structured* ADES variant (observations grouped in
    /// `<obsBlock>` elements with an `<obsContext>`) and the *flat* variant
    /// (all `<optical>` elements at the document root) are supported.
    ///
    /// # Observer resolution
    ///
    /// Each observation is associated with an [`crate::observer::dataset::ObserverId::MpcCode`]
    /// derived from the `<stn>` field (flat ADES) or the `<mpcCode>` field of
    /// the enclosing `<obsContext>` (structured ADES).  The full observatory
    /// metadata is resolved lazily from the MPC catalogue on the first call to
    /// [`ObsDataset::get_observer`].
    ///
    /// # Uncertainty resolution (per observation, in priority order)
    ///
    /// 1. `<rmsRA>` / `<rmsDec>` — statistical 1-σ uncertainties in arcseconds.
    /// 2. `<precRA>` / `<precDec>` — precision-based uncertainties in arcseconds.
    /// 3. `error_ra` / `error_dec` — caller-supplied fallbacks in arcseconds.
    ///
    /// If none of the three sources is available for a given observation the
    /// function **panics**.
    ///
    /// # Arguments
    ///
    /// - `ades_path` — path to the ADES XML file to read.
    /// - `error_ra`  — optional fallback 1-σ RA uncertainty in **arcseconds**.
    /// - `error_dec` — optional fallback 1-σ Dec uncertainty in **arcseconds**.
    ///
    /// # Panics
    ///
    /// - If the file cannot be read.
    /// - If the XML cannot be parsed as either ADES variant.
    /// - If a required uncertainty value is missing and no fallback was given.
    pub fn from_ades(
        ades_path: &Utf8Path,
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> Self {
        parse_ades_file(ades_path, error_ra, error_dec)
    }
}
