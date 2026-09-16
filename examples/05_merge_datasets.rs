//! Merge datasets with `merge_from`, and see why loading files
//! *independently* is a common way to trigger an `ObsId` collision.
//!
//! Requires the `mpc_80_col` feature:
//!
//! ```sh
//! cargo run --example 05_merge_datasets --features mpc_80_col
//! ```

use camino::Utf8PathBuf;
use photom::observation_dataset::{ObsDataset, ObsDatasetError};

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. `from_mpc_80_col` always numbers observations 0..N *within a single
    //    file*. Loading two files this way and merging them by hand collides
    //    on every id they have in common — this is the pitfall, not an edge
    //    case: it happens on any two files loaded independently.
    let ds_8467 = ObsDataset::from_mpc_80_col(data("8467.obs"))?;
    let ds_33803 = ObsDataset::from_mpc_80_col(data("33803.obs"))?;
    println!(
        "8467.obs: {} obs | 33803.obs: {} obs",
        ds_8467.observation_count(),
        ds_33803.observation_count()
    );

    match ds_8467.clone().merge_from(ds_33803.clone()) {
        Ok(_) => unreachable!("both files start at ObsId 0, a collision is expected"),
        Err(ObsDatasetError::DuplicateObsIds(ids)) => {
            println!(
                "\nnaive merge rejected as expected: {} colliding ObsId(s), e.g. {:?}",
                ids.len(),
                &ids[..ids.len().min(3)]
            );
        }
        Err(other) => return Err(other.into()),
    }

    // 2. The fix: `extend_from_mpc_80_col` (and its sibling
    //    `from_mpc_80_col_files`) reads each new file with a starting id
    //    offset by the current dataset size, so ids never collide, and
    //    merges automatically.
    let (extended, warnings) = ds_8467.clone().extend_from_mpc_80_col(&[data("33803.obs")]);
    assert!(warnings.is_empty());
    println!(
        "\nextend_from_mpc_80_col succeeded: {} + {} = {} observations",
        ds_8467.observation_count(),
        ds_33803.observation_count(),
        extended.observation_count()
    );

    Ok(())
}
