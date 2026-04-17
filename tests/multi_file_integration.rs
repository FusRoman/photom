//! Integration tests for multi-file loading (builder, from_*_files, extend_from_*).

#![cfg(any(feature = "ades", feature = "mpc_80_col"))]

use camino::Utf8Path;

fn data(name: &str) -> camino::Utf8PathBuf {
    camino::Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

// ── MPC 80-column multi-file tests ──────────────────────────────────────────

#[cfg(feature = "mpc_80_col")]
mod mpc_multi_file {
    use super::*;
    use photom::{TrajId, observation_dataset::ObsDataset};

    /// Loading two different MPC files should sum their observation counts.
    #[test]
    fn from_mpc_80_col_files_sums_observation_counts() {
        let p1 = data("8467.obs");
        let p2 = data("33803.obs");
        let (ds, errors) = ObsDataset::from_mpc_80_col_files(&[p1.as_path(), p2.as_path()]);
        assert!(errors.is_empty(), "no errors expected: {errors:?}");

        let count_1 = ObsDataset::from_mpc_80_col(p1.as_path())
            .unwrap()
            .observation_count();
        let count_2 = ObsDataset::from_mpc_80_col(p2.as_path())
            .unwrap()
            .observation_count();
        assert_eq!(ds.observation_count(), count_1 + count_2);
    }

    /// Trajectories from both files should appear in the merged dataset.
    #[test]
    fn from_mpc_80_col_files_contains_both_trajectories() {
        let p1 = data("8467.obs");
        let p2 = data("33803.obs");
        let (ds, _) = ObsDataset::from_mpc_80_col_files(&[p1.as_path(), p2.as_path()]);
        assert!(ds.len_trajectory(&TrajId::Int(8467)).is_some());
        assert!(ds.len_trajectory(&TrajId::Int(33803)).is_some());
    }

    /// Passing the same file twice doubles the observation count and merges
    /// both halves under the same TrajId.
    #[test]
    fn from_mpc_80_col_files_same_traj_id_merges() {
        let p = data("8467.obs");
        let single_count = ObsDataset::from_mpc_80_col(p.as_path())
            .unwrap()
            .observation_count();

        let (ds, errors) = ObsDataset::from_mpc_80_col_files(&[p.as_path(), p.as_path()]);
        assert!(errors.is_empty());
        assert_eq!(ds.observation_count(), single_count * 2);
        // Both halves are merged under TrajId::Int(8467).
        assert_eq!(
            ds.len_trajectory(&TrajId::Int(8467)).unwrap_or(0),
            single_count * 2
        );
    }

    /// A non-existent file produces a warning and is skipped.
    #[test]
    fn from_mpc_80_col_files_skips_missing_file() {
        let good = data("8467.obs");
        let bad = Utf8Path::new("/no/such/file.obs");
        let (ds, errors) = ObsDataset::from_mpc_80_col_files(&[bad, good.as_path()]);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one error for missing file"
        );
        let good_count = ObsDataset::from_mpc_80_col(good.as_path())
            .unwrap()
            .observation_count();
        assert_eq!(ds.observation_count(), good_count);
    }

    /// `extend_from_mpc_80_col` grows an existing dataset.
    #[test]
    fn extend_from_mpc_80_col_increases_count() {
        let p1 = data("8467.obs");
        let p2 = data("33803.obs");
        let mut ds = ObsDataset::from_mpc_80_col(p1.as_path()).unwrap();
        let before = ds.observation_count();
        let errors = ds.extend_from_mpc_80_col(&[p2.as_path()]);
        assert!(errors.is_empty());
        assert!(ds.observation_count() > before);
    }
}

// ── ADES multi-file tests ────────────────────────────────────────────────────

#[cfg(feature = "ades")]
mod ades_multi_file {
    use super::*;
    use photom::observation_dataset::ObsDataset;

    /// Loading two ADES files should sum their observation counts.
    #[test]
    fn from_ades_files_sums_observation_counts() {
        let p1 = data("example_ades.xml");
        let p2 = data("example_ades2.xml");
        let (ds, errors) = ObsDataset::from_ades_files(&[p1.as_path(), p2.as_path()], None, None);
        assert!(errors.is_empty(), "no errors expected: {errors:?}");

        let c1 = ObsDataset::from_ades(p1.as_path(), None, None)
            .unwrap()
            .observation_count();
        let c2 = ObsDataset::from_ades(p2.as_path(), None, None)
            .unwrap()
            .observation_count();
        assert_eq!(ds.observation_count(), c1 + c2);
    }

    /// A non-existent ADES file produces a warning and is skipped.
    #[test]
    fn from_ades_files_skips_missing_file() {
        let good = data("example_ades.xml");
        let bad = Utf8Path::new("/no/such/file.xml");
        let (ds, errors) = ObsDataset::from_ades_files(&[bad, good.as_path()], None, None);
        assert_eq!(errors.len(), 1);
        let good_count = ObsDataset::from_ades(good.as_path(), None, None)
            .unwrap()
            .observation_count();
        assert_eq!(ds.observation_count(), good_count);
    }

    /// `extend_from_ades` grows an existing dataset.
    #[test]
    fn extend_from_ades_increases_count() {
        let p1 = data("example_ades.xml");
        let p2 = data("example_ades2.xml");
        let mut ds = ObsDataset::from_ades(p1.as_path(), None, None).unwrap();
        let before = ds.observation_count();
        let errors = ds.extend_from_ades(&[p2.as_path()], None, None);
        assert!(errors.is_empty());
        assert!(ds.observation_count() > before);
    }
}

// ── Builder tests ────────────────────────────────────────────────────────────

#[cfg(all(feature = "ades", feature = "mpc_80_col"))]
mod builder_tests {
    use super::*;
    use photom::observation_dataset::ObsDataset;
    use photom::observation_dataset::builder::ObsDatasetBuilder;

    /// Builder with no files produces an empty dataset and no warnings.
    #[test]
    fn builder_empty_produces_empty_dataset() {
        let (ds, warnings) = ObsDatasetBuilder::new().build();
        assert_eq!(ds.observation_count(), 0);
        assert!(warnings.is_empty());
    }

    /// Builder from an existing dataset preserves the original observations.
    #[test]
    fn builder_from_dataset_preserves_observations() {
        let p = data("8467.obs");
        let existing = ObsDataset::from_mpc_80_col(p.as_path()).unwrap();
        let count = existing.observation_count();
        let (ds, warnings) = ObsDatasetBuilder::from_dataset(existing).build();
        assert_eq!(ds.observation_count(), count);
        assert!(warnings.is_empty());
    }

    /// Builder can load an MPC file and an ADES file and merge them.
    #[test]
    fn builder_add_mpc_and_ades_merges() {
        let mpc = data("8467.obs");
        let ades = data("example_ades.xml");
        let mpc_count = ObsDataset::from_mpc_80_col(mpc.as_path())
            .unwrap()
            .observation_count();
        let ades_count = ObsDataset::from_ades(ades.as_path(), None, None)
            .unwrap()
            .observation_count();

        let (ds, warnings) = ObsDatasetBuilder::new()
            .add_mpc_80_col(&[mpc.as_path()])
            .add_ades(&[ades.as_path()], None, None)
            .build();

        assert!(warnings.is_empty());
        assert_eq!(ds.observation_count(), mpc_count + ades_count);
    }

    /// A bad file produces a warning; the rest of the load continues.
    #[test]
    fn builder_bad_file_produces_warning() {
        let good = data("8467.obs");
        let bad = Utf8Path::new("/no/such.obs");
        let (ds, warnings) = ObsDatasetBuilder::new()
            .add_mpc_80_col(&[bad, good.as_path()])
            .build();

        assert_eq!(warnings.len(), 1);
        let good_count = ObsDataset::from_mpc_80_col(good.as_path())
            .unwrap()
            .observation_count();
        assert_eq!(ds.observation_count(), good_count);
    }
}
