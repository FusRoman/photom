#![cfg(feature = "datafusion")]

use crate::io::datafusion::input_uri::InputUri;
use crate::io::datafusion::loader::{LoadObsArgs, LoadObsError, load_obs_sync};
use crate::observation_dataset::ObsDataset;

impl ObsDataset {
    /// Construct an [`ObsDataset`] from a Parquet file at the given URI.
    ///
    /// Supported URI schemes: `file://`, `http://`, `https://`, `hdfs://`.
    /// The URI is resolved to an object-store backend via the
    /// [`crate::io::datafusion::storage`] module, the Parquet file is
    /// scanned with DataFusion, and the resulting Arrow
    /// [`RecordBatch`](arrow_array::RecordBatch)es are converted into an
    /// `ObsDataset`.
    ///
    /// This method blocks the calling thread internally using a single-threaded
    /// Tokio runtime.  Use [`crate::io::datafusion::load_obs_from_parquet_uri`]
    /// in async contexts.
    ///
    /// # Arguments
    ///
    /// - `uri` — URI string pointing to the Parquet resource.
    /// - `args` — configuration for the ingestion pipeline (error model, LRU
    ///   cache size).  Use [`LoadObsArgs::default`] for sensible defaults.
    ///
    /// # Errors
    ///
    /// Returns a [`LoadObsError`] if the URI is invalid, the resource is not
    /// found, DataFusion fails to read the file, or a required column is
    /// absent or has an incorrect type.
    pub fn from_parquet_uri(uri: &str, args: LoadObsArgs) -> Result<Self, LoadObsError> {
        let input = InputUri(uri.to_owned());
        load_obs_sync(&input, args)
    }
}
