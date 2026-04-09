//! Photometric measurement types used throughout the pipeline.
//!
//! This module defines the data structures that encode the photometric
//! information attached to each [`crate::observation::Observation`]:
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

/// The photometric bandpass filter used during an observation.
///
/// Source catalogues encode filter information in two ways: some use
/// human-readable strings such as `"V"`, `"r'"`, or `"Gaia-G"`, while others
/// store an integer filter code.  This enum accommodates both representations
/// without loss of information.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Photometry {
    /// Apparent magnitude of the detected source.
    pub magnitude: f64,
    /// 1-σ uncertainty on the magnitude, expressed in the same unit as
    /// [`magnitude`](Self::magnitude).
    pub error: f64,
    /// Bandpass filter through which the measurement was taken.
    pub filter: Filter,
}
