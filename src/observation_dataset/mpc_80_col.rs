#![cfg(feature = "mpc_80_col")]

use camino::Utf8Path;

use crate::{io::mpc_80_col::parse_mpc_80_col_file, observation_dataset::ObsDataset};

impl ObsDataset {
    /// Build an [`ObsDataset`] by reading an **MPC 80-column** observation file.
    ///
    /// The format is the fixed-width ASCII format distributed by the Minor
    /// Planet Center: each observation occupies exactly 80 columns.  Both
    /// numbered objects (columns 1–5) and provisionally designated objects
    /// (columns 6–12) are supported, and a single file may contain
    /// observations for **multiple trajectories**.
    ///
    /// ## Field layout (1-indexed, MPC convention)
    ///
    /// | Columns | Content                          |
    /// |---------|----------------------------------|
    /// | 1–5     | Minor planet number              |
    /// | 6–12    | Provisional designation          |
    /// | 13      | Discovery flag                   |
    /// | 14–15   | Note 1, Note 2 (observation type)|
    /// | 16–32   | Date `YYYY MM DD.ddddd` (UTC)    |
    /// | 33–44   | RA `HH MM SS.sss` (J2000)        |
    /// | 45–56   | Dec `±DD MM SS.ss` (J2000)       |
    /// | 66–71   | Magnitude                        |
    /// | 72      | Photometric band                 |
    /// | 78–80   | MPC observatory code             |
    ///
    /// ## Trajectory identifiers
    ///
    /// The trajectory ID is derived per-line: if the minor planet number
    /// (columns 1–5, stripped of leading zeros) is non-empty it is used;
    /// otherwise the provisional designation (columns 6–12) is used.  If the
    /// resulting string parses as an integer it becomes a `TrajId::Int`;
    /// otherwise a `TrajId::Str`.
    ///
    /// ## Time and angle conventions
    ///
    /// - Dates are given in UTC and internally converted to **MJD (TT)** via
    ///   [`hifitime`].
    /// - RA/Dec are stored in **radians**; uncertainties are derived from the
    ///   number of decimal places in the seconds field (RA: `10⁻ⁿ × 15`
    ///   arcsec, Dec: `10⁻ⁿ` arcsec).
    ///
    /// ## Skipped lines
    ///
    /// - Lines shorter than 80 bytes are silently ignored.
    /// - Lines where column 15 (0-indexed: 14) equals `'s'` are secondary
    ///   satellite-position lines and are silently ignored.
    ///
    /// ## Panics
    ///
    /// Panics if the file cannot be read, or if a line cannot be parsed
    /// (consistent with the project's fail-fast policy for corrupted inputs).
    pub fn from_mpc_80_col(path: &Utf8Path) -> ObsDataset {
        parse_mpc_80_col_file(path)
    }
}
