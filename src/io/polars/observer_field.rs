// ── per-row observer resolution ───────────────────────────────────────────────

use crate::{
    io::polars::error::PolarsError,
    observer::{Observer, dataset::ObserverId, mpc::MpcCode},
};

/// All nullable observer inputs for one row, already extracted from the
/// [`ChunkedArray`](polars::prelude::ChunkedArray)s.
///
/// This struct is a plain data carrier used exclusively by
/// [`resolve_observer`].  Each field corresponds to one observer column in
/// the source [`DataFrame`]; a value of `None` means the cell was `null` or
/// the column was absent from the frame.
///
/// The lifetime `'df` is tied to the source [`DataFrame`]: the `mpc_code`
/// field borrows string data directly from Polars' internal memory so that no
/// per-row heap allocation is needed for the common case of an absent or null
/// MPC code.
pub(crate) struct RawObsRow<'df> {
    /// Geodetic longitude in degrees east of Greenwich (`obs_lon` column).
    pub(crate) obs_lon: Option<f64>,
    /// Geodetic latitude in degrees (`obs_lat` column).
    pub(crate) obs_lat: Option<f64>,
    /// Altitude above the reference ellipsoid in meters (`obs_alt` column).
    pub(crate) obs_alt: Option<f64>,
    /// Right ascension measurement accuracy in radians (`obs_ra_acc` column).
    pub(crate) obs_ra_acc: Option<f64>,
    /// Declination measurement accuracy in radians (`obs_dec_acc` column).
    pub(crate) obs_dec_acc: Option<f64>,
    /// Raw MPC observatory code string (`mpc_code_obs` column).
    ///
    /// Borrowed from Polars' internal memory; no heap allocation is performed
    /// unless the code is actually non-null and valid.
    pub(crate) mpc_code: Option<&'df str>,
}

/// Result of resolving one row's observer fields into a typed observer
/// identity.
///
/// [`resolve_observer`] returns one of these three variants depending on
/// which observer columns were populated for the row.
pub enum ResolvedObserver {
    /// A fully specified geodetic observer that was constructed from the
    /// `obs_lon` / `obs_lat` / `obs_alt` / `obs_ra_acc` / `obs_dec_acc`
    /// columns.
    ///
    /// The inner [`Observer`] should be interned into the dataset's
    /// `custom_observers` list so that identical sites share a single slot.
    Geodetic(Observer),

    /// An MPC code reference resolved from the `mpc_code_obs` column.
    ///
    /// Astrometric accuracy for MPC observers is not stored here; it is
    /// looked up from the error model at query time.
    Mpc(ObserverId),

    /// No observer information was available for this row (all observer
    /// columns were `null` or absent).
    None,
}

/// Resolve the observer for a single row according to the column precedence
/// rules.
///
/// The resolution is applied in the following order:
///
/// 1. If `mpc_code` is non-null it takes precedence over the geodetic triplet
///    regardless of whether the triplet columns are also populated.
/// 2. If `mpc_code` is null and the geodetic triplet (`obs_lon`, `obs_lat`,
///    `obs_alt`) is fully non-null, a custom [`Observer`] is constructed.
///    `obs_ra_acc` and `obs_dec_acc` must also be non-null in this case.
/// 3. If `mpc_code` is null and the geodetic triplet is entirely null (or all
///    three columns were absent from the frame), the observer is
///    [`ResolvedObserver::None`].
/// 4. A partially-null geodetic triplet — where one or two of the three
///    columns are non-null — is always an error regardless of the other
///    fields.
///
/// # Arguments
///
/// - `row`     — the nullable observer values for this row.
/// - `row_idx` — zero-based row index, used in error messages.
///
/// # Returns
///
/// A [`ResolvedObserver`] variant describing the observer for this row.
///
/// # Errors
///
/// - [`PolarsError::InvalidMpcCode`] if `mpc_code` is non-null but is not
///   exactly three ASCII bytes.
/// - [`PolarsError::MissingAccuracyForGeodesic`] if the geodetic triplet is
///   fully non-null but `obs_ra_acc` or `obs_dec_acc` is `null`.
/// - [`PolarsError::PartialTripletNull`] if exactly one or two of the three
///   geodetic columns are non-null.
/// - [`PolarsError::DataConversionError`] if [`Observer::new`] rejects the
///   coordinate values (e.g. a `NaN` was encountered).
pub(crate) fn resolve_observer(
    row: &RawObsRow<'_>,
    row_idx: usize,
) -> Result<ResolvedObserver, PolarsError> {
    // ── precedence: MPC code wins when non-null ──────────────────────────────
    if let Some(code_str) = row.mpc_code {
        let bytes: MpcCode = code_str
            .as_bytes()
            .try_into()
            .map_err(|_| PolarsError::InvalidMpcCode(code_str.to_owned(), row_idx))?;
        return Ok(ResolvedObserver::Mpc(ObserverId::MpcCode(bytes)));
    }

    // ── geodetic triplet ─────────────────────────────────────────────────────
    let lon_present = row.obs_lon.is_some();
    let lat_present = row.obs_lat.is_some();
    let alt_present = row.obs_alt.is_some();

    match (lon_present, lat_present, alt_present) {
        // All three present → build Observer.
        (true, true, true) => {
            let obs_ra_acc = row
                .obs_ra_acc
                .ok_or(PolarsError::MissingAccuracyForGeodesic(row_idx))?;
            let obs_dec_acc = row
                .obs_dec_acc
                .ok_or(PolarsError::MissingAccuracyForGeodesic(row_idx))?;

            let observer = Observer::new(
                row.obs_lon.unwrap(),
                row.obs_lat.unwrap(),
                row.obs_alt.unwrap(),
                None,
                Some(obs_ra_acc),
                Some(obs_dec_acc),
            )
            .map_err(|e| PolarsError::DataConversionError(e.to_string()))?;

            Ok(ResolvedObserver::Geodetic(observer))
        }

        // All three absent → no observer.
        (false, false, false) => Ok(ResolvedObserver::None),

        // Partial null → error.
        _ => Err(PolarsError::PartialTripletNull {
            row: row_idx,
            obs_lon: lon_present,
            obs_lat: lat_present,
            obs_alt: alt_present,
        }),
    }
}
