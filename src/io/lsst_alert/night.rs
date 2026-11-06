//! Night identification for Rubin alerts.
//!
//! Rubin encodes the observing night in the `visit` field:
//! `visit = dayObs × 100000 + seqNum`, where `dayObs` is the `YYYYMMDD`
//! calendar date in the UTC−12 h time zone.  The date therefore rolls over at
//! 12:00 UTC (morning in Chile) and **a night is never split in two**.
//!
//! photom's [`NightId`] for an LSST alert is the MJD of the `dayObs` date
//! (the date of the evening).  For an instant `t` this equals
//! $\lfloor \mathrm{MJD}(t) - 0.5 \rfloor$, which [`night_from_mjd`]
//! computes so that the two sources can be cross-checked.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`VISIT_SEQ_FACTOR`] | const | `100000`, the `seqNum` span inside `visit` |
//! | [`day_obs_from_visit`] | fn | `visit` → `dayObs` (`YYYYMMDD`) |
//! | [`day_obs_to_night_id`] | fn | `dayObs` → [`NightId`] (MJD of the date) |
//! | [`night_from_mjd`] | fn | MJD instant → night number `⌊mjd − 0.5⌋` |

use crate::NightId;

use super::error::LsstAlertError;

/// Multiplier separating `dayObs` from `seqNum` inside a Rubin `visit`.
pub const VISIT_SEQ_FACTOR: i64 = 100_000;

/// Julian Day Number of MJD 0 (1858-11-17), plus one because the JDN of a
/// civil date refers to its noon while MJD 0 starts at its midnight.
const JDN_OF_MJD_ZERO: i64 = 2_400_001;

/// Extract the `dayObs` date from a Rubin `visit` identifier.
///
/// # Arguments
/// - `visit` — Rubin visit id, `dayObs × 100000 + seqNum`.
///
/// # Returns
/// `dayObs` as an integer `YYYYMMDD`.  The date itself is **not** validated
/// here; [`day_obs_to_night_id`] does that.
///
/// # Errors
/// [`LsstAlertError::InvalidVisit`] when `visit` is not positive or when the
/// quotient does not fit an eight-digit date.
pub fn day_obs_from_visit(visit: i64) -> Result<u32, LsstAlertError> {
    if visit <= 0 {
        return Err(LsstAlertError::InvalidVisit(visit));
    }
    u32::try_from(visit / VISIT_SEQ_FACTOR)
        .ok()
        .filter(|d| (10_000_101..=99_991_231).contains(d))
        .ok_or(LsstAlertError::InvalidVisit(visit))
}

/// Convert a `YYYYMMDD` date into the [`NightId`] equal to its MJD.
///
/// The civil date → Julian Day Number conversion is the integer algorithm of
/// Fliegel & Van Flandern (1968), valid for every Gregorian date of interest.
///
/// # Arguments
/// - `day_obs` — calendar date encoded as `YYYYMMDD`.
///
/// # Returns
/// `NightId(MJD at 00:00 UTC of that date)`, e.g. `20000101` → `NightId(51544)`.
///
/// # Errors
/// [`LsstAlertError::InvalidDayObs`] when the month or day is invalid or the
/// date precedes 1858-11-17 (negative MJD).
pub fn day_obs_to_night_id(day_obs: u32) -> Result<NightId, LsstAlertError> {
    let invalid = || LsstAlertError::InvalidDayObs(day_obs);
    let year = i64::from(day_obs / 10_000);
    let month = i64::from(day_obs / 100 % 100);
    let day = i64::from(day_obs % 100);
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return Err(invalid());
    }

    // Fliegel & Van Flandern; Rust's `/` truncates toward zero as required.
    let a = (month - 14) / 12;
    let jdn = (1461 * (year + 4800 + a)) / 4 + (367 * (month - 2 - 12 * a)) / 12
        - (3 * ((year + 4900 + a) / 100)) / 4
        + day
        - 32_075;

    u32::try_from(jdn - JDN_OF_MJD_ZERO)
        .map(NightId)
        .map_err(|_| invalid())
}

/// Night number of an instant: the MJD of the evening date (rollover at noon UTC).
///
/// # Arguments
/// - `mjd` — instant as a Modified Julian Date.
///
/// # Returns
/// $\lfloor \mathrm{mjd} - 0.5 \rfloor$ as a signed integer (it is not
/// clamped, so it can be compared with a [`NightId`] even for odd inputs).
#[inline]
pub fn night_from_mjd(mjd: f64) -> i64 {
    (mjd - 0.5).floor() as i64
}

/// Number of days in `month` of the Gregorian `year`.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn real_visit_to_night() {
        let day_obs = day_obs_from_visit(2_026_103_000_508).unwrap();
        assert_eq!(day_obs, 20_261_030);
        assert_eq!(day_obs_to_night_id(day_obs), Ok(NightId(61_343)));
    }

    #[test]
    fn reference_dates() {
        assert_eq!(day_obs_to_night_id(18_581_117), Ok(NightId(0)));
        assert_eq!(day_obs_to_night_id(20_000_101), Ok(NightId(51_544)));
        assert_eq!(day_obs_to_night_id(20_000_229), Ok(NightId(51_603)));
        assert_eq!(day_obs_to_night_id(20_240_301), Ok(NightId(60_370)));
    }

    #[test]
    fn invalid_dates_are_rejected() {
        for bad in [
            20_261_301, 20_261_000, 20_261_100, 20_261_131, 20_260_229, 19_000_229, 18_581_116,
        ] {
            assert_eq!(
                day_obs_to_night_id(bad),
                Err(LsstAlertError::InvalidDayObs(bad)),
                "{bad} must be rejected"
            );
        }
    }

    #[test]
    fn invalid_visits_are_rejected() {
        for bad in [0, -2_026_103_000_508, 42, i64::MAX] {
            assert_eq!(
                day_obs_from_visit(bad),
                Err(LsstAlertError::InvalidVisit(bad))
            );
        }
    }

    #[test]
    fn night_rolls_over_at_noon_utc() {
        // 2026-10-30 23:00 UTC and 2026-10-31 11:59 UTC belong to night 61343.
        assert_eq!(night_from_mjd(61_343.0 + 23.0 / 24.0), 61_343);
        assert_eq!(night_from_mjd(61_344.0 + 11.98 / 24.0), 61_343);
        assert_eq!(night_from_mjd(61_344.5), 61_344);
    }

    proptest! {
        /// Consecutive days have consecutive NightIds (no gaps, no repeats).
        #[test]
        fn prop_consecutive_days(year in 1900u32..2200, month in 1u32..=12) {
            let n_days = days_in_month(i64::from(year), i64::from(month)) as u32;
            for day in 1..n_days {
                let a = day_obs_to_night_id(year * 10_000 + month * 100 + day).unwrap();
                let b = day_obs_to_night_id(year * 10_000 + month * 100 + day + 1).unwrap();
                prop_assert_eq!(b.0, a.0 + 1);
            }
        }
    }
}
