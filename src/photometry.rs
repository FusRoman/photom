//! Photometric measurement types used throughout the pipeline.
//!
//! This module defines the data structures that encode the photometric
//! information attached to each [`crate::observation_dataset::observation::Observation`]:
//!
//! - [`Filter`] — the bandpass filter through which a source was observed,
//!   represented either as a human-readable label or as a numeric code.
//! - [`Photometry`] — the apparent magnitude of a detected source together
//!   with its 1-σ uncertainty and the filter used.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`Filter`] | enum | Photometric bandpass filter identifier |
//! | [`Photometry`] | struct | Apparent magnitude, uncertainty, and filter |

use std::fmt;

/// The photometric bandpass filter used during an observation.
///
/// Source catalogues encode filter information in two ways: some use
/// human-readable strings such as `"V"`, `"r'"`, or `"Gaia-G"`, while others
/// store an integer filter code.  This enum accommodates both representations
/// without loss of information.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Filter {
    /// A human-readable filter label (e.g. `"V"`, `"r'"`, `"Gaia-G"`).
    String(String),
    /// A numeric filter code used when the source catalogue encodes filters as integers.
    Int(u32),
}

/// A photometric measurement attached to a single observation.
///
/// `Photometry` bundles the apparent magnitude of a detected source, its
/// 1-σ measurement uncertainty, and the bandpass filter through which the
/// observation was taken.  All magnitude values follow the standard
/// astronomical convention (lower value = brighter source).
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Photometry {
    /// Apparent magnitude of the detected source.
    pub magnitude: f64,
    /// 1-σ uncertainty on the magnitude, expressed in the same unit as
    /// [`magnitude`](Self::magnitude).
    pub error: f64,
    /// Bandpass filter through which the measurement was taken.
    pub filter: Filter,
}

// ---------------------------------------------------------------------------
// Filter Display
// ---------------------------------------------------------------------------

/// Formats the filter as its label or numeric code.
///
/// # Format
///
/// - `Filter::String(s)` → `s`
/// - `Filter::Int(n)`    → `n`
impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Filter::String(s) => write!(f, "{s}"),
            Filter::Int(n) => write!(f, "{n}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Photometry Display
// ---------------------------------------------------------------------------

/// Formats the photometric measurement as a compact inline string.
///
/// # Format
///
/// ```text
/// 18.432 ± 0.023 mag [r']
/// ```
impl fmt::Display for Photometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.3} ± {:.3} mag [{}]",
            self.magnitude, self.error, self.filter
        )
    }
}

#[cfg(test)]
mod photometry_tests {
    use super::*;

    // ------------------------------------------------------------------
    // Filter
    // ------------------------------------------------------------------

    #[test]
    fn filter_string_eq() {
        let a = Filter::String("V".to_string());
        let b = Filter::String("V".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn filter_int_eq() {
        assert_eq!(Filter::Int(3), Filter::Int(3));
    }

    #[test]
    fn filter_variants_ne() {
        assert_ne!(
            Filter::String("V".to_string()),
            Filter::String("r".to_string())
        );
        assert_ne!(Filter::Int(1), Filter::Int(2));
    }

    #[test]
    fn filter_clone() {
        let f = Filter::String("Gaia-G".to_string());
        assert_eq!(f.clone(), f);
    }

    #[test]
    fn filter_ord() {
        // Int(1) < Int(2)
        assert!(Filter::Int(1) < Filter::Int(2));
        // String("B") < String("V") lexicographically
        assert!(Filter::String("B".to_string()) < Filter::String("V".to_string()));
    }

    // ------------------------------------------------------------------
    // Photometry
    // ------------------------------------------------------------------

    #[test]
    fn photometry_fields_accessible() {
        let p = Photometry {
            magnitude: 18.5,
            error: 0.05,
            filter: Filter::String("r".to_string()),
        };
        assert_eq!(p.magnitude, 18.5);
        assert_eq!(p.error, 0.05);
        assert_eq!(p.filter, Filter::String("r".to_string()));
    }

    #[test]
    fn photometry_clone_eq() {
        let p = Photometry {
            magnitude: 15.0,
            error: 0.1,
            filter: Filter::Int(7),
        };
        assert_eq!(p.clone(), p);
    }

    #[test]
    fn photometry_partial_ord() {
        let brighter = Photometry {
            magnitude: 12.0,
            error: 0.1,
            filter: Filter::String("V".to_string()),
        };
        let fainter = Photometry {
            magnitude: 20.0,
            error: 0.1,
            filter: Filter::String("V".to_string()),
        };
        assert!(brighter < fainter);
    }

    #[test]
    fn photometry_debug() {
        let p = Photometry {
            magnitude: 1.0,
            error: 0.0,
            filter: Filter::Int(0),
        };
        let s = format!("{:?}", p);
        assert!(s.contains("magnitude"));
    }
}
