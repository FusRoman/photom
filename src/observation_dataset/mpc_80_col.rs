#![cfg(feature = "mpc_80_col")]

use camino::{Utf8Path, Utf8PathBuf};

use crate::{
    ObsDatasetBuilder, TrajId,
    io::mpc_80_col::{Mpc80ColError, parse_mpc_80_col_file},
    observation_dataset::ObsDataset,
};

impl ObsDataset {
    /// Register an alternate designation that resolves to `primary`.
    ///
    /// Intended for use by ingestion backends; not part of the public API.
    pub(crate) fn register_alias(&mut self, alias: String, primary: TrajId) {
        self.index.register_alias(alias, primary);
    }

    /// Build an [`ObsDataset`] by reading an **MPC 80-column** observation file.
    ///
    /// The format is the fixed-width ASCII format distributed by the Minor
    /// Planet Center: each observation occupies exactly 80 columns.  Both
    /// numbered objects (columns 1–5) and provisionally designated objects
    /// (columns 6–12) are supported, and a single file may contain
    /// observations for **multiple trajectories**.
    ///
    /// ## Errors
    ///
    /// Returns [`Mpc80ColError::Io`] if the file cannot be read, or
    /// [`Mpc80ColError::InvalidLine`] if a line cannot be parsed.
    pub fn from_mpc_80_col(path: &Utf8Path) -> Result<ObsDataset, Mpc80ColError> {
        parse_mpc_80_col_file(path)
    }

    /// Build an [`ObsDataset`] from **multiple** MPC 80-column files.
    ///
    /// Files that cannot be parsed are skipped; their errors are collected in
    /// the second element of the returned tuple.
    ///
    /// # Arguments
    ///
    /// - `paths` — slice of paths to MPC 80-column files to load.
    pub fn from_mpc_80_col_files(paths: &[&Utf8Path]) -> (Self, Vec<(Utf8PathBuf, Mpc80ColError)>) {
        let mut dataset: Option<ObsDataset> = None;
        let mut errors: Vec<(Utf8PathBuf, Mpc80ColError)> = Vec::new();

        for &path in paths {
            match parse_mpc_80_col_file(path) {
                Ok(other) => {
                    if let Some(ref mut ds) = dataset {
                        ds.merge_from_unchecked(other);
                    } else {
                        dataset = Some(other);
                    }
                }
                Err(e) => errors.push((path.to_owned(), e)),
            }
        }

        let ds = dataset.unwrap_or_else(ObsDataset::empty);
        (ds, errors)
    }

    /// Merge observations from **multiple** MPC 80-column files into `self`.
    ///
    /// Files that cannot be parsed are skipped; their errors are returned.
    ///
    /// # Arguments
    ///
    /// - `paths` — slice of paths to MPC 80-column files.
    pub fn extend_from_mpc_80_col(
        &mut self,
        paths: &[&Utf8Path],
    ) -> Vec<(Utf8PathBuf, Mpc80ColError)> {
        let mut errors: Vec<(Utf8PathBuf, Mpc80ColError)> = Vec::new();
        for &path in paths {
            match parse_mpc_80_col_file(path) {
                Ok(other) => self.merge_from_unchecked(other),
                Err(e) => errors.push((path.to_owned(), e)),
            }
        }
        errors
    }
}

impl ObsDatasetBuilder {
    /// Load one or more MPC 80-column files and merge their observations.
    ///
    /// Files that raise a [`Mpc80ColError`]
    /// are skipped and appended to the internal warning list.
    ///
    /// # Arguments
    ///
    /// - `paths` — slice of paths to MPC 80-column observation files to load.
    pub fn add_mpc_80_col(mut self, paths: &[&Utf8Path]) -> Self {
        for &path in paths {
            match crate::io::mpc_80_col::parse_mpc_80_col_file(path) {
                Ok(other) => {
                    if let Some(ref mut ds) = self.dataset {
                        ds.merge_from_unchecked(other);
                    } else {
                        self.dataset = Some(other);
                    }
                }
                Err(error) => {
                    use crate::LoadWarning;

                    self.warnings.push(LoadWarning::MpcFile {
                        path: path.to_owned(),
                        error,
                    });
                }
            }
        }
        self
    }
}
