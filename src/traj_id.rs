// ── TrajId ────────────────────────────────────────────────────────────────────

use std::{
    fmt::{self, Display},
    hash::Hash,
};

use ahash::RandomState;

/// Typed identifier for a single trajectory.
///
/// A trajectory can be identified either by a **32-bit unsigned integer** (e.g.
/// a running index or a catalogue number) or by a **string label** (e.g. a
/// Minor Planet Center provisional designation such as `"2020 AV2"`, or a
/// proper name such as `"Ceres"`).
///
/// The column type of `traj_id` in the source `DataFrame` determines which
/// variant is used: a `UInt32` column produces [`TrajId::Int`] keys and a
/// `String` column produces [`TrajId::Str`] keys.  Mixing both types in a
/// single dataset is not supported; the column must be uniformly one type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrajId {
    /// A 32-bit unsigned integer identifier (e.g. a catalogue number).
    Int(u32),
    /// A string label (e.g. a MPC provisional designation or a proper name).
    Str(String),
}

impl TrajId {
    /// Derive a deterministic `u64` hash for per-trajectory RNG seeding.
    ///
    /// Uses [`ahash`] with fixed seeds to guarantee cross-run determinism.
    /// Combined with a caller-supplied base seed, this ensures that each
    /// trajectory always receives the same noise sequence regardless of
    /// processing order.
    pub fn stable_hash(&self) -> u64 {
        RandomState::with_seeds(1, 2, 3, 4).hash_one(self)
    }
}

impl Display for TrajId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrajId::Int(n) => write!(f, "{}", n),
            TrajId::Str(s) => write!(f, "{}", s),
        }
    }
}

// --- Infallible conversions (enable `.into()` directly) ----------------------

impl From<&TrajId> for TrajId {
    #[inline]
    fn from(id: &TrajId) -> Self {
        id.clone()
    }
}

impl From<u32> for TrajId {
    #[inline]
    fn from(n: u32) -> Self {
        TrajId::Int(n)
    }
}

impl From<u16> for TrajId {
    #[inline]
    fn from(n: u16) -> Self {
        TrajId::Int(n as u32)
    }
}

impl From<u8> for TrajId {
    #[inline]
    fn from(n: u8) -> Self {
        TrajId::Int(n as u32)
    }
}

impl From<&u32> for TrajId {
    /// Convenience to allow `(&n).into()` without dereferencing at call sites.
    #[inline]
    fn from(n: &u32) -> Self {
        TrajId::Int(*n)
    }
}

impl From<&u16> for TrajId {
    /// Convenience to allow `(&n).into()` without dereferencing at call sites.
    #[inline]
    fn from(n: &u16) -> Self {
        TrajId::Int(*n as u32)
    }
}

impl From<&u8> for TrajId {
    /// Convenience to allow `(&n).into()` without dereferencing at call sites.
    #[inline]
    fn from(n: &u8) -> Self {
        TrajId::Int(*n as u32)
    }
}

impl From<String> for TrajId {
    #[inline]
    fn from(s: String) -> Self {
        TrajId::Str(s)
    }
}

impl From<&String> for TrajId {
    /// Clones the string to build a `String`-backed identifier.
    #[inline]
    fn from(s: &String) -> Self {
        TrajId::Str(s.clone())
    }
}

impl From<&str> for TrajId {
    /// Note: this **does not** parse numeric strings into `Int`. Use `FromStr` if you want
    /// `"1234"` to become `TrajId::Int(1234)`.
    #[inline]
    fn from(s: &str) -> Self {
        TrajId::Str(s.to_string())
    }
}

// --- Fallible conversions (use `.try_into()` to be overflow-safe) ------------

impl TryFrom<usize> for TrajId {
    type Error = std::num::TryFromIntError;

    /// Convert a `usize` into `Int(u32)` if it fits.
    #[inline]
    fn try_from(n: usize) -> Result<Self, Self::Error> {
        Ok(TrajId::Int(u32::try_from(n)?))
    }
}

impl TryFrom<u64> for TrajId {
    type Error = std::num::TryFromIntError;

    /// Convert a `u64` into `Int(u32)` if it fits.
    #[inline]
    fn try_from(n: u64) -> Result<Self, Self::Error> {
        Ok(TrajId::Int(u32::try_from(n)?))
    }
}

impl TryFrom<i64> for TrajId {
    type Error = &'static str;

    /// Convert a non-negative `i64` into `Int(u32)` if it fits.
    #[inline]
    fn try_from(n: i64) -> Result<Self, Self::Error> {
        if n < 0 {
            return Err("negative value is not a valid TrajId::Int");
        }
        let n = u64::try_from(n).map_err(|_| "conversion failed")?;
        let n = u32::try_from(n).map_err(|_| "value exceeds u32 range")?;
        Ok(TrajId::Int(n))
    }
}

// --- Smart parsing from &str via `FromStr` (optional) ------------------------

impl std::str::FromStr for TrajId {
    type Err = std::num::ParseIntError;

    /// Try to parse a `TrajId` from a string.
    ///
    /// Rules
    /// -----
    /// - Pure digits that fit in `u32` → `Int(u32)`.
    /// - Otherwise                         → `String(String)`.
    ///
    /// Note
    /// ----
    /// If the string is *only* digits but **does not** fit in `u32`, this returns the
    /// original `ParseIntError`. If you prefer to always fallback to `String` on
    /// overflow, we can change the policy (but it’s usually better to fail loudly).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.parse::<u32>() {
            Ok(n) => Ok(TrajId::Int(n)),
            Err(e) => {
                if s.chars().any(|c| !c.is_ascii_digit()) {
                    Ok(TrajId::Str(s.to_string()))
                } else {
                    Err(e)
                }
            }
        }
    }
}

#[cfg(test)]
mod traj_id_tests {
    use super::*;

    /// Same input always produces the same hash (deterministic within a run).
    #[test]
    fn test_stable_hash_deterministic() {
        let id_int = TrajId::Int(42);
        let id_str = TrajId::Str("2023 AB1".to_string());

        assert_eq!(id_int.stable_hash(), id_int.stable_hash());
        assert_eq!(id_str.stable_hash(), id_str.stable_hash());
    }

    /// Two distinct ids must not collide.
    #[test]
    fn test_stable_hash_distinct_inputs() {
        let a = TrajId::Int(0);
        let b = TrajId::Int(1);
        let c = TrajId::Str("0".to_string());

        assert_ne!(a.stable_hash(), b.stable_hash());
        assert_ne!(a.stable_hash(), c.stable_hash());
    }

    /// The hash is cross-run stable: expected values are hard-coded from a
    /// reference run and must never change without a deliberate version bump.
    #[test]
    fn test_stable_hash_cross_run_stability() {
        let expected_int = TrajId::Int(42).stable_hash();
        let expected_str = TrajId::Str("2023 AB1".to_string()).stable_hash();

        assert_eq!(14966747408011497582, expected_int);
        assert_eq!(16188224256132921782, expected_str);
    }

    /// The XOR combination used for RNG seeding does not collapse to zero
    /// for any tested input (base_seed ^ stable_hash() != 0).
    #[test]
    fn test_stable_hash_xor_seed_nonzero() {
        let base_seed: u64 = 0xdeadbeefcafebabe;
        let ids = [
            TrajId::Int(0),
            TrajId::Int(1),
            TrajId::Int(u32::MAX),
            TrajId::Str(String::new()),
            TrajId::Str("test".to_string()),
        ];
        for id in &ids {
            assert_ne!(base_seed ^ id.stable_hash(), 0);
        }
    }

    /// Cloned ids produce the same hash.
    #[test]
    fn test_stable_hash_clone_equality() {
        let id = TrajId::Str("C/2024 X1".to_string());
        assert_eq!(id.stable_hash(), id.clone().stable_hash());
    }
}
