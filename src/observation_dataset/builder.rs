//! Builder for constructing an [`ObsDataset`] from multiple source files.
//!
//! [`ObsDatasetBuilder`] lets you accumulate observations from any number of
//! ADES XML or MPC 80-column files in a single pass.  Files that cannot be
//! parsed are **skipped** and recorded as [`LoadWarning`] values; the build
//! never fails due to a single bad file.
//!
//! # Example
//!
//! ```rust,ignore
//! use camino::Utf8Path;
//! use photom::observation_dataset::builder::ObsDatasetBuilder;
//!
//! let (dataset, warnings) = ObsDatasetBuilder::new()
//!     .add_ades(&[Utf8Path::new("a.xml"), Utf8Path::new("b.xml")], None, None)
//!     .add_mpc_80_col(&[Utf8Path::new("c.obs")])
//!     .build();
//!
//! for w in &warnings {
//!     eprintln!("skipped: {w}");
//! }
//! println!("{} observations loaded", dataset.observation_count());
//! ```

use std::fmt;

use crate::observation_dataset::ObsDataset;

// camino is only available when ades or mpc_80_col is enabled.
#[cfg(any(feature = "ades", feature = "mpc_80_col"))]
use camino::Utf8PathBuf;

// ── LoadWarning ──────────────────────────────────────────────────────────────

/// A file that could not be parsed during a multi-file load.
///
/// Produced by [`ObsDatasetBuilder::build`] (and by the shortcut methods
/// `from_ades_files`, `extend_from_ades`, etc.) when an individual source
/// file raises a parse error.  The file is skipped and the rest of the load
/// continues.
#[derive(Debug)]
pub enum LoadWarning {
    /// An ADES XML file could not be parsed.
    #[cfg(feature = "ades")]
    AdesFile {
        /// Path to the file that failed.
        path: Utf8PathBuf,
        /// The error that was raised.
        error: crate::io::ades::AdesError,
    },
    /// An MPC 80-column file could not be parsed.
    #[cfg(feature = "mpc_80_col")]
    MpcFile {
        /// Path to the file that failed.
        path: Utf8PathBuf,
        /// The error that was raised.
        error: crate::io::mpc_80_col::Mpc80ColError,
    },
}

impl fmt::Display for LoadWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "ades")]
            LoadWarning::AdesFile { path, error } => {
                write!(f, "ADES file '{path}' skipped: {error}")
            }
            #[cfg(feature = "mpc_80_col")]
            LoadWarning::MpcFile { path, error } => {
                write!(f, "MPC 80-col file '{path}' skipped: {error}")
            }
            // Unreachable when neither feature is enabled (enum has no variants),
            // but we still need a valid arm that uses `f` to silence the
            // unused-variable warning the compiler emits in that configuration.
            #[allow(unreachable_patterns)]
            _ => write!(f, ""),
        }
    }
}

// ── ObsDatasetBuilder ────────────────────────────────────────────────────────

/// Accumulates observations from multiple source files into a single [`ObsDataset`].
///
/// Files that fail to parse are recorded as [`LoadWarning`] values and
/// skipped; the builder never aborts early.  Call [`build`](Self::build) to
/// consume the builder and obtain the final dataset together with any
/// accumulated warnings.
#[derive(Default)]
pub struct ObsDatasetBuilder {
    pub(crate) dataset: Option<ObsDataset>,
    pub(crate) warnings: Vec<LoadWarning>,
}

impl ObsDatasetBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialise the builder from an existing dataset.
    ///
    /// Additional files added with [`add_ades`](Self::add_ades) or
    /// [`add_mpc_80_col`](Self::add_mpc_80_col) will be merged into this
    /// dataset.
    pub fn from_dataset(dataset: ObsDataset) -> Self {
        Self {
            dataset: Some(dataset),
            warnings: Vec::new(),
        }
    }

    /// Consume the builder, returning the accumulated dataset and any warnings.
    ///
    /// If no files were loaded successfully, the returned dataset is empty
    /// (zero observations).
    pub fn build(self) -> (ObsDataset, Vec<LoadWarning>) {
        let dataset = self.dataset.unwrap_or_else(empty_dataset);
        (dataset, self.warnings)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn empty_dataset() -> ObsDataset {
    ObsDataset::new(vec![], vec![], None, None, None, None)
}
