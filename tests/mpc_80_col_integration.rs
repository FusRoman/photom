//! Integration tests for the MPC 80-column reader.
#![cfg(feature = "mpc_80_col")]

use photom::{TrajId, observation_dataset::ObsDataset};

fn data(name: &str) -> camino::Utf8PathBuf {
    camino::Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

// ---------------------------------------------------------------------------
// 8467.obs — numbered asteroid, single trajectory, 61 observations
// ---------------------------------------------------------------------------

#[test]
fn test_8467_obs_count() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    assert_eq!(ds.observation_count(), 61);
}

#[test]
fn test_8467_single_trajectory() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    let count = ds.iter_traj_id().map(|it| it.count()).unwrap_or(0);
    assert_eq!(count, 1);
}

#[test]
fn test_8467_traj_id_is_integer() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    assert_eq!(ds.len_trajectory(TrajId::Int(8467)).unwrap_or(0), 61);
}

#[test]
fn test_8467_mjd_positive() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    for obs in ds.iter_observations() {
        assert!(
            obs.mjd_tt() > 0.0,
            "MJD should be positive: {}",
            obs.mjd_tt()
        );
    }
}

#[test]
fn test_8467_ra_in_range() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    use std::f64::consts::PI;
    for obs in ds.iter_observations() {
        let ra = obs.equ_coord().ra;
        assert!((0.0..=2.0 * PI).contains(&ra), "RA out of range: {ra}");
    }
}

#[test]
fn test_8467_dec_in_range() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    use std::f64::consts::FRAC_PI_2;
    for obs in ds.iter_observations() {
        let dec = obs.equ_coord().dec;
        assert!(
            (-FRAC_PI_2..=FRAC_PI_2).contains(&dec),
            "Dec out of range: {dec}"
        );
    }
}

#[test]
fn test_8467_uncertainties_positive() {
    let ds = ObsDataset::from_mpc_80_col(data("8467.obs").as_path()).unwrap();
    for obs in ds.iter_observations() {
        let coord = obs.equ_coord();
        assert!(coord.ra_error > 0.0, "RA uncertainty should be > 0");
        assert!(coord.dec_error > 0.0, "Dec uncertainty should be > 0");
    }
}

// ---------------------------------------------------------------------------
// K25D50B.obs — provisionally designated object, single trajectory
// ---------------------------------------------------------------------------

#[test]
fn test_k25d50b_obs_count() {
    let ds = ObsDataset::from_mpc_80_col(data("K25D50B.obs").as_path()).unwrap();
    assert_eq!(ds.observation_count(), 20);
}

#[test]
fn test_k25d50b_traj_id_is_string() {
    let ds = ObsDataset::from_mpc_80_col(data("K25D50B.obs").as_path()).unwrap();
    assert_eq!(
        ds.len_trajectory(TrajId::Str("K25D50B".to_string()))
            .unwrap_or(0),
        20
    );
}

// ---------------------------------------------------------------------------
// 2015AB.obs — provisionally designated object with multiple designations, single trajectory
//   Primary : K09R05F (first designation encountered in the file)
//   Alias   : K15A00B
//   Total   : 37 observations (all under the primary TrajId)
// ---------------------------------------------------------------------------

#[test]
fn test_2015ab_single_trajectory() {
    let ds = ObsDataset::from_mpc_80_col(data("2015AB.obs").as_path()).unwrap();
    let count = ds.iter_traj_id().map(|it| it.count()).unwrap_or(0);
    assert_eq!(count, 1, "2015AB.obs must produce a single trajectory");
}

#[test]
fn test_2015ab_total_obs_count() {
    let ds = ObsDataset::from_mpc_80_col(data("2015AB.obs").as_path()).unwrap();
    assert_eq!(ds.observation_count(), 37);
}

#[test]
fn test_2015ab_primary_traj_id() {
    let ds = ObsDataset::from_mpc_80_col(data("2015AB.obs").as_path()).unwrap();
    // All observations are under the primary TrajId K09R05F.
    assert_eq!(
        ds.len_trajectory(TrajId::Str("K09R05F".to_string()))
            .unwrap_or(0),
        37
    );
}

#[test]
fn test_2015ab_k15a00b_is_alias() {
    let ds = ObsDataset::from_mpc_80_col(data("2015AB.obs").as_path()).unwrap();
    // K15A00B must resolve to the primary K09R05F.
    let resolved = ds.resolve_alias("K15A00B");
    assert_eq!(
        resolved,
        Some(&TrajId::Str("K09R05F".to_string())),
        "K15A00B must resolve to K09R05F"
    );
}

#[test]
fn test_2015ab_k09r05f_not_alias() {
    let ds = ObsDataset::from_mpc_80_col(data("2015AB.obs").as_path()).unwrap();
    // K09R05F is the primary and should not be an alias.
    assert!(
        ds.resolve_alias("K09R05F").is_none(),
        "K09R05F is the primary and should not be an alias"
    );
}

// ---------------------------------------------------------------------------
// 33803.obs — numbered asteroid with many stations
// ---------------------------------------------------------------------------

#[test]
fn test_33803_obs_count() {
    let ds = ObsDataset::from_mpc_80_col(data("33803.obs").as_path()).unwrap();
    assert_eq!(ds.observation_count(), 129);
}

#[test]
fn test_33803_single_trajectory() {
    let ds = ObsDataset::from_mpc_80_col(data("33803.obs").as_path()).unwrap();
    let count = ds.iter_traj_id().map(|it| it.count()).unwrap_or(0);
    assert_eq!(count, 1);
}

#[test]
fn test_33803_traj_id() {
    let ds = ObsDataset::from_mpc_80_col(data("33803.obs").as_path()).unwrap();
    assert_eq!(ds.len_trajectory(TrajId::Int(33803)).unwrap_or(0), 129);
}

#[test]
fn test_33803_uncertainties_positive() {
    let ds = ObsDataset::from_mpc_80_col(data("33803.obs").as_path()).unwrap();
    for obs in ds.iter_observations() {
        let coord = obs.equ_coord();
        assert!(coord.ra_error > 0.0, "RA uncertainty should be > 0");
        assert!(coord.dec_error > 0.0, "Dec uncertainty should be > 0");
    }
}
