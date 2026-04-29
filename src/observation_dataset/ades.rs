#![cfg(feature = "ades")]

use camino::{Utf8Path, Utf8PathBuf};

use crate::{
    Arcseconds, ObsDatasetBuilder,
    io::ades::{AdesError, parse_ades_file},
    observation_dataset::ObsDataset,
};

impl ObsDataset {
    /// Build an [`ObsDataset`] by reading an ADES XML file.
    ///
    /// # Arguments
    ///
    /// - `ades_path` — path to the ADES XML file to read.
    /// - `error_ra`  — optional fallback 1-σ RA uncertainty in **arcseconds**.
    /// - `error_dec` — optional fallback 1-σ Dec uncertainty in **arcseconds**.
    ///
    /// # Errors
    ///
    /// Returns [`AdesError::Io`] if the file cannot be read,
    /// [`AdesError::ParseXml`] if the XML cannot be parsed, or
    /// [`AdesError::MissingTrajId`] / [`AdesError::MissingRaError`] /
    /// [`AdesError::MissingDecError`] if a required field is absent.
    pub fn from_ades(
        ades_path: impl AsRef<Utf8Path>,
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> Result<Self, AdesError> {
        parse_ades_file(ades_path.as_ref(), error_ra, error_dec, 0)
    }

    /// Build an [`ObsDataset`] from **multiple** ADES XML files.
    ///
    /// Files that cannot be parsed are skipped; their errors are collected in
    /// the second element of the returned tuple.
    ///
    /// # Arguments
    ///
    /// - `paths`     — slice of paths to ADES XML files to load.
    /// - `error_ra`  — optional fallback RA uncertainty (arcseconds).
    /// - `error_dec` — optional fallback Dec uncertainty (arcseconds).
    pub fn from_ades_files<P: AsRef<Utf8Path>>(
        paths: &[P],
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> (Self, Vec<(Utf8PathBuf, AdesError)>) {
        let mut dataset: Option<ObsDataset> = None;
        let mut errors: Vec<(Utf8PathBuf, AdesError)> = Vec::new();

        for path in paths {
            let start_id = dataset
                .as_ref()
                .map_or(0, |ds| ds.observation_count() as u64);
            match parse_ades_file(path.as_ref(), error_ra, error_dec, start_id) {
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
                Err(e) => errors.push((path.as_ref().to_owned(), e)),
            }
        }

        let ds = dataset.unwrap_or_else(ObsDataset::empty);
        (ds, errors)
    }

    /// Merge observations from **multiple** ADES XML files into `self`.
    ///
    /// Files that cannot be parsed are skipped; their errors are returned.
    ///
    /// # Arguments
    ///
    /// - `paths`     — slice of paths to ADES XML files.
    /// - `error_ra`  — optional fallback RA uncertainty (arcseconds).
    /// - `error_dec` — optional fallback Dec uncertainty (arcseconds).
    pub fn extend_from_ades<P: AsRef<Utf8Path>>(
        mut self,
        paths: &[P],
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> (Self, Vec<(Utf8PathBuf, AdesError)>) {
        let mut errors: Vec<(Utf8PathBuf, AdesError)> = Vec::new();
        for path in paths {
            let start_id = self.observation_count() as u64;
            match parse_ades_file(path.as_ref(), error_ra, error_dec, start_id) {
                // IDs are globally unique: merge_from cannot fail here.
                Ok(other) => {
                    self = self
                        .merge_from(other)
                        .expect("IDs are globally unique: merge_from cannot fail");
                }
                Err(e) => errors.push((path.as_ref().to_owned(), e)),
            }
        }
        (self, errors)
    }
}

impl ObsDatasetBuilder {
    /// Load one or more ADES XML files and merge their observations.
    ///
    /// Files that raise an [`AdesError`] are
    /// skipped and appended to the internal warning list.
    ///
    /// # Arguments
    ///
    /// - `paths`     — slice of paths to ADES XML files to load.
    /// - `error_ra`  — optional fallback RA uncertainty (arcseconds).
    /// - `error_dec` — optional fallback Dec uncertainty (arcseconds).
    pub fn add_ades<P: AsRef<Utf8Path>>(
        mut self,
        paths: &[P],
        error_ra: Option<crate::Arcseconds>,
        error_dec: Option<crate::Arcseconds>,
    ) -> Self {
        for path in paths {
            let start_id = self
                .dataset
                .as_ref()
                .map_or(0, |ds| ds.observation_count() as u64);
            match crate::io::ades::parse_ades_file(path.as_ref(), error_ra, error_dec, start_id) {
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

                    self.warnings.push(LoadWarning::AdesFile {
                        path: path.as_ref().to_owned(),
                        error,
                    });
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod ades_index_consistency_tests {
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

    /// Loading a single ADES file satisfies the index-consistency invariant.
    #[test]
    fn index_consistency_from_ades() {
        let path_str = fixture("example_ades.xml");
        let ds = ObsDataset::from_ades(path_str, None, None)
            .expect("example_ades.xml must parse without error");
        assert_index_consistency(&ds);
    }

    /// Loading multiple ADES files satisfies the index-consistency invariant.
    #[test]
    fn index_consistency_from_ades_files() {
        let path1_str = fixture("example_ades.xml");
        let path2_str = fixture("example_ades2.xml");
        let (ds, errors) = ObsDataset::from_ades_files(&[path1_str, path2_str], None, None);
        assert!(
            errors.is_empty(),
            "no parse errors expected loading both ades fixtures, got: {errors:?}"
        );
        assert_index_consistency(&ds);
    }

    /// Extending a dataset with ADES files satisfies the index-consistency invariant.
    #[test]
    fn index_consistency_extend_from_ades() {
        let path1_str = fixture("example_ades.xml");
        let path2_str = fixture("example_ades2.xml");
        let base = ObsDataset::from_ades(path1_str, None, None)
            .expect("example_ades.xml must parse without error");
        let (extended, errors) = base.extend_from_ades(&[path2_str], None, None);
        assert!(
            errors.is_empty(),
            "no parse errors expected when extending with ades fixture, got: {errors:?}"
        );
        assert_index_consistency(&extended);
    }
}
