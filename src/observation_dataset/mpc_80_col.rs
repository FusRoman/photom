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
        parse_mpc_80_col_file(path, 0)
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
            let start_id = dataset
                .as_ref()
                .map_or(0, |ds| ds.observation_count() as u64);
            match parse_mpc_80_col_file(path, start_id) {
                Ok(other) => {
                    if let Some(ds) = dataset.take() {
                        // IDs are globally unique: merge_from cannot fail here.
                        dataset = Some(
                            ds.merge_from(other)
                                .expect("IDs are globally unique: merge_from cannot fail"),
                        );
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
        mut self,
        paths: &[&Utf8Path],
    ) -> (Self, Vec<(Utf8PathBuf, Mpc80ColError)>) {
        let mut errors: Vec<(Utf8PathBuf, Mpc80ColError)> = Vec::new();
        for &path in paths {
            let start_id = self.observation_count() as u64;
            match parse_mpc_80_col_file(path, start_id) {
                // IDs are globally unique: merge_from cannot fail here.
                Ok(other) => {
                    self = self
                        .merge_from(other)
                        .expect("IDs are globally unique: merge_from cannot fail");
                }
                Err(e) => errors.push((path.to_owned(), e)),
            }
        }
        (self, errors)
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
            let start_id = self
                .dataset
                .as_ref()
                .map_or(0, |ds| ds.observation_count() as u64);
            match crate::io::mpc_80_col::parse_mpc_80_col_file(path, start_id) {
                Ok(other) => {
                    if let Some(ds) = self.dataset.take() {
                        // IDs are globally unique: merge_from cannot fail here.
                        self.dataset = Some(
                            ds.merge_from(other)
                                .expect("IDs are globally unique: merge_from cannot fail"),
                        );
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

#[cfg(test)]
mod mpc_80_col_index_consistency_tests {
    use camino::Utf8Path;

    use crate::observation_dataset::ObsDataset;

    fn assert_index_consistency(dataset: &ObsDataset) {
        for (idx, obs) in dataset.iter_observations().enumerate() {
            assert_eq!(
                idx,
                obs.index(),
                "index-consistency violated: enumeration position {idx} != obs.index() {}",
                obs.index()
            );
        }
    }

    fn fixture(name: &str) -> String {
        format!("{}/tests/data/{}", env!("CARGO_MANIFEST_DIR"), name)
    }

    /// Loading a single MPC 80-column file satisfies the index-consistency invariant.
    #[test]
    fn index_consistency_from_mpc_80_col() {
        let path_str = fixture("8467.obs");
        let path = Utf8Path::new(&path_str);
        let ds = ObsDataset::from_mpc_80_col(path).expect("8467.obs must parse without error");
        assert_index_consistency(&ds);
    }

    /// Loading multiple MPC 80-column files satisfies the index-consistency invariant.
    #[test]
    fn index_consistency_from_mpc_80_col_files() {
        let path1_str = fixture("8467.obs");
        let path2_str = fixture("33803.obs");
        let path1 = Utf8Path::new(&path1_str);
        let path2 = Utf8Path::new(&path2_str);
        let (ds, errors) = ObsDataset::from_mpc_80_col_files(&[path1, path2]);
        assert!(
            errors.is_empty(),
            "no parse errors expected loading mpc fixtures, got: {errors:?}"
        );
        assert_index_consistency(&ds);
    }

    /// Extending a dataset with MPC 80-column files satisfies the index-consistency invariant.
    #[test]
    fn index_consistency_extend_from_mpc_80_col() {
        let path1_str = fixture("8467.obs");
        let path2_str = fixture("33803.obs");
        let path1 = Utf8Path::new(&path1_str);
        let path2 = Utf8Path::new(&path2_str);
        let base = ObsDataset::from_mpc_80_col(path1).expect("8467.obs must parse without error");
        let (extended, errors) = base.extend_from_mpc_80_col(&[path2]);
        assert!(
            errors.is_empty(),
            "no parse errors expected when extending with mpc fixture, got: {errors:?}"
        );
        assert_index_consistency(&extended);
    }
}
