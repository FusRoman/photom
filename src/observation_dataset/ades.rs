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
        ades_path: &Utf8Path,
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> Result<Self, AdesError> {
        parse_ades_file(ades_path, error_ra, error_dec)
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
    pub fn from_ades_files(
        paths: &[&Utf8Path],
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> (Self, Vec<(Utf8PathBuf, AdesError)>) {
        let mut dataset: Option<ObsDataset> = None;
        let mut errors: Vec<(Utf8PathBuf, AdesError)> = Vec::new();

        for &path in paths {
            match parse_ades_file(path, error_ra, error_dec) {
                Ok(other) => {
                    if let Some(ref mut ds) = dataset {
                        ds.merge_from(other);
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

    /// Merge observations from **multiple** ADES XML files into `self`.
    ///
    /// Files that cannot be parsed are skipped; their errors are returned.
    ///
    /// # Arguments
    ///
    /// - `paths`     — slice of paths to ADES XML files.
    /// - `error_ra`  — optional fallback RA uncertainty (arcseconds).
    /// - `error_dec` — optional fallback Dec uncertainty (arcseconds).
    pub fn extend_from_ades(
        &mut self,
        paths: &[&Utf8Path],
        error_ra: Option<Arcseconds>,
        error_dec: Option<Arcseconds>,
    ) -> Vec<(Utf8PathBuf, AdesError)> {
        let mut errors: Vec<(Utf8PathBuf, AdesError)> = Vec::new();
        for &path in paths {
            match parse_ades_file(path, error_ra, error_dec) {
                Ok(other) => self.merge_from(other),
                Err(e) => errors.push((path.to_owned(), e)),
            }
        }
        errors
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
    pub fn add_ades(
        mut self,
        paths: &[&Utf8Path],
        error_ra: Option<crate::Arcseconds>,
        error_dec: Option<crate::Arcseconds>,
    ) -> Self {
        for &path in paths {
            match crate::io::ades::parse_ades_file(path, error_ra, error_dec) {
                Ok(other) => {
                    if let Some(ref mut ds) = self.dataset {
                        ds.merge_from(other);
                    } else {
                        self.dataset = Some(other);
                    }
                }
                Err(error) => {
                    use crate::LoadWarning;

                    self.warnings.push(LoadWarning::AdesFile {
                        path: path.to_owned(),
                        error,
                    });
                }
            }
        }
        self
    }
}
