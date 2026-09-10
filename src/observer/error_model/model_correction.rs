use crate::{
    constants::ARCSEC_TO_RAD,
    observation_dataset::{ObsDataset, index::ObsMapIndex},
    observer::{
        dataset::ObserverId,
        error_model::{ObsErrorModel, get_bias_rms},
    },
};

impl ObsDataset {
    /// Set the astrometric error model used for MPC observatory initialisation.
    /// This method allows changing the error model after the dataset has been constructed,
    /// which will affect the accuracies assigned to MPC-coded observers when the MPC table is loaded.
    ///
    /// Note that if the MPC table has already been initialised,
    /// changing the error model will not retroactively update the observer accuracies;
    /// the new error model will only take effect on the first call to `mpc_observers()`
    /// if the MPC table has not yet been loaded.
    ///
    /// # Arguments
    ///
    /// - `error_model` — the new [`ObsErrorModel`] to use for MPC observatory initialisation.
    pub fn set_error_model(&mut self, error_model: ObsErrorModel) {
        self.observer_dataset.mpc_error_model = Some(error_model);
    }

    /// Consume `self`, attach an astrometric error model, and return the updated dataset.
    ///
    /// This is the chainable counterpart of [`ObsDataset::set_error_model`]:
    /// it allows the error model to be set in a builder-style pipeline without
    /// requiring a separate `let mut` binding.
    ///
    /// # Arguments
    ///
    /// - `error_model` — the [`ObsErrorModel`] variant to store in the dataset.
    ///
    /// # Returns
    ///
    /// The same dataset with the error model set.
    pub fn with_error_model(mut self, error_model: ObsErrorModel) -> Self {
        self.observer_dataset.mpc_error_model = Some(error_model);
        self
    }

    /// Get a reference to the currently attached astrometric error model, if any.
    ///
    /// # Returns
    ///
    /// - `Some(&ObsErrorModel)` if an error model is attached to the dataset,
    /// - `None` if no error model is attached.
    pub fn get_error_model(&self) -> Option<&ObsErrorModel> {
        self.observer_dataset.mpc_error_model.as_ref()
    }
}

pub trait ModelCorrection {
    /// Apply the stored astrometric error model to each observation's uncertainties.
    ///
    /// For every observation whose observer is identified by an MPC code, the
    /// method looks up the RMS values `(rms_ra, rms_dec)` from the model data
    /// and replaces the stored uncertainties with the element-wise maximum of
    /// the existing value and the model-derived value:
    ///
    /// $$
    /// \sigma_\alpha = \max\!\left(\sigma_{\alpha,\text{fmt}},\;
    ///     \frac{\sigma_{\alpha,\text{model}}}{\cos\delta}\right), \qquad
    /// \sigma_\delta = \max\!\left(\sigma_{\delta,\text{fmt}},\;
    ///     \sigma_{\delta,\text{model}}\right)
    /// $$
    ///
    /// where $\sigma_{\alpha,\text{model}}$ and $\sigma_{\delta,\text{model}}$
    /// are the model RMS values converted to radians, and $\delta$ is the
    /// declination of the observation.
    ///
    /// Observations with no MPC observer, or whose code is not found in the
    /// model, are left unchanged.
    ///
    /// If no error model is stored in the dataset, or the model file cannot be
    /// read, `self` is returned unmodified.
    ///
    /// # Note
    ///
    /// The catalog code used for the model lookup is always `"c"`.  Per-site
    /// catalog-code handling is not yet implemented.
    ///
    /// # Returns
    ///
    /// The updated dataset with corrected observation uncertainties.
    fn apply_model_errors(self) -> Self;

    /// Apply a batch RMS correction to the astrometric uncertainties of each observation.
    ///
    /// Astrometric errors of several observations of the **same object** taken in
    /// quick succession from the **same site** are strongly correlated: they
    /// share the same field, the same astrometric plate solution and the same
    /// reference stars. Treating such observations as statistically independent
    /// overstates the amount of information they carry. This method compensates
    /// for that by inflating the reported `ra_error` and `dec_error` of every
    /// observation in a batch by a factor derived from the batch size.
    ///
    /// ### Grouping
    ///
    /// Batches are formed **within a single trajectory** — never across
    /// trajectories. Two observations belong to the same batch if and only if
    /// all of the following hold:
    ///
    /// - they belong to the **same trajectory** (same key in the trajectory
    ///   index), **and**
    /// - they share the same `observer` identity, **and**
    /// - once the trajectory's observations are ordered by epoch, the time gap
    ///   between the two consecutive observations is not greater than `gap_max`.
    ///
    /// The batch size `n` used in the correction factor is therefore the number
    /// of observations *of that one object*, from one site, within one
    /// `gap_max` window. It does **not** depend on how many unrelated
    /// trajectories happen to be loaded in the same dataset, nor on how many
    /// other observations the site recorded in the same night.
    ///
    /// An observation that is listed under more than one trajectory (for example
    /// when trajectories are competing linkage hypotheses) is assigned to a
    /// single batch and corrected exactly once.
    ///
    /// ### Fallback when no trajectory index is present
    ///
    /// The correction requires the trajectory index (`traj_id`) to know which
    /// observations belong to the same object. When the dataset was built
    /// without it, that information is unavailable and there is no safe way to
    /// form batches: every observation is treated as its own batch of size 1,
    /// which makes the correction a **no-op** (factor `sqrt(1) = 1`). Attach a
    /// trajectory index at ingestion time to enable the correction.
    ///
    /// ### Correction factor
    ///
    /// For a batch of size $n$:
    ///
    /// | Model    | Condition | Factor                   |
    /// |----------|-----------|--------------------------|
    /// | `FCCT14` | any $n$   | $\sqrt{n}$               |
    /// | `VFCC17` | $n \geq 5$| $\sqrt{n \times 0.25}$   |
    /// | `VFCC17` | $n < 5$   | $\sqrt{n}$               |
    ///
    /// The `VFCC17` branch encodes the fact that the correlation saturates:
    /// beyond a handful of observations, adding more does not keep reducing the
    /// effective information proportionally.
    ///
    /// Both `ra_error` and `dec_error` are multiplied by the same factor:
    ///
    /// $$\sigma' = \sigma \times \text{factor}(n)$$
    ///
    /// A batch of size 1 always yields factor 1 and leaves its observation
    /// untouched.
    ///
    /// # Arguments
    ///
    /// - `gap_max` – Maximum time gap (days) between two consecutive observations
    ///   of the same object and observer for them to be considered part of the
    ///   same batch. A typical value is $8/24 \approx 0.333$ days (8 hours).
    ///
    /// # Returns
    ///
    /// The dataset with corrected uncertainties. Consumed by value and returned
    /// by value (builder pattern).
    ///
    /// If no error model is attached to the dataset, `self` is returned unmodified.
    /// Attach an error model first via `with_error_model` or `set_error_model`.
    ///
    /// # Notes
    ///
    /// - The internal observation order is **not** modified. Grouping is performed
    ///   on a sorted index without mutating the observation vector.
    /// - Time comparisons are based on Modified Julian Date in Terrestrial Time
    ///   (`MJD TT`). Uncertainties are expressed in **radians**.
    fn apply_batch_rms_correction(self, gap_max: f64) -> ObsDataset;
}

impl ModelCorrection for ObsDataset {
    fn apply_model_errors(mut self) -> Self {
        let model_data = match &self.observer_dataset.mpc_error_model {
            Some(em) => match em.read_error_model_file() {
                Ok(data) => data,
                Err(_) => return self,
            },
            None => return self,
        };

        for obs in &mut self.observations {
            let mpc_code = match obs.observer {
                Some(ObserverId::MpcCode(code)) => code,
                _ => continue,
            };

            if let Some((rms_ra, rms_dec)) = get_bias_rms(&model_data, mpc_code, "c") {
                let cos_dec = obs.equ_coord.dec.cos();

                let model_ra_rad = rms_ra as f64 * ARCSEC_TO_RAD / cos_dec;
                let model_dec_rad = rms_dec as f64 * ARCSEC_TO_RAD;

                obs.equ_coord.ra_error = obs.equ_coord.ra_error.max(model_ra_rad);
                obs.equ_coord.dec_error = obs.equ_coord.dec_error.max(model_dec_rad);
            }
        }

        self
    }

    fn apply_batch_rms_correction(mut self, gap_max: f64) -> Self {
        let error_model = match self.observer_dataset.mpc_error_model {
            Some(em) => em,
            None => return self,
        };

        let n_obs = self.observations.len();
        if n_obs == 0 {
            return self;
        }

        // The correction needs to know which observations belong to the same
        // object. Without a trajectory index that information is missing, so
        // batching cannot be done safely and the method degrades to a no-op.
        let Some(traj_map) = self.index.obs_index_by_trajectory.as_ref() else {
            return self;
        };

        // ── Assign every observation to a correlation group ──────────────────
        //
        // A "group" is the set of observations that are *allowed* to share a
        // batch, i.e. the observations of one trajectory. Batches are then
        // formed strictly inside a group by the (observer, gap_max) rule; they
        // never straddle a group boundary.
        //
        // Group ids in `[0, n_obs)` are singleton groups (one per observation);
        // an observation keeps its singleton id unless the trajectory index
        // places it in a shared group. Shared group ids start at `n_obs` so
        // they can never collide with a singleton id. An observation listed
        // under several trajectories keeps the last shared id written and is
        // consequently corrected exactly once.
        let mut group_of: Vec<usize> = (0..n_obs).collect();
        for (traj_rank, entry) in traj_map.values().enumerate() {
            let group_id = n_obs + traj_rank;
            match entry {
                ObsMapIndex::Contiguous { start, end } => {
                    for slot in &mut group_of[*start..*end] {
                        *slot = group_id;
                    }
                }
                ObsMapIndex::Split(indices) => {
                    for &idx in indices {
                        group_of[idx] = group_id;
                    }
                }
            }
        }

        // Single allocation: sort indices by (group, observer, mjd_tt) so that
        // every batch is a contiguous slice of `sorted_indices`.
        let mut sorted_indices: Vec<usize> = (0..n_obs).collect();
        sorted_indices.sort_unstable_by(|&a, &b| {
            let oa = &self.observations[a];
            let ob = &self.observations[b];
            group_of[a]
                .cmp(&group_of[b])
                .then_with(|| oa.observer.cmp(&ob.observer))
                .then_with(|| oa.mjd_tt.partial_cmp(&ob.mjd_tt).unwrap())
        });

        // ── Walk the sorted indices, flushing one batch at a time ────────────
        //
        // A batch ends at `j` when the run of observations that started at
        // `batch_start` cannot be extended: end of data, a change of group, a
        // change of observer, or a time gap greater than `gap_max`.
        let mut batch_start = 0usize;
        for j in 1..=n_obs {
            let boundary = j == n_obs || {
                let prev = &self.observations[sorted_indices[j - 1]];
                let curr = &self.observations[sorted_indices[j]];
                group_of[sorted_indices[j - 1]] != group_of[sorted_indices[j]]
                    || prev.observer != curr.observer
                    || (curr.mjd_tt - prev.mjd_tt) > gap_max
            };

            if !boundary {
                continue;
            }

            let n = j - batch_start;
            let factor = match error_model {
                ObsErrorModel::VFCC17 if n >= 5 => (n as f64 * 0.25).sqrt(),
                _ => (n as f64).sqrt(),
            };
            // Skip the write-back for size-1 batches (factor == 1.0), which are
            // the overwhelming majority once grouping is per-object.
            if factor != 1.0 {
                for &idx in &sorted_indices[batch_start..j] {
                    self.observations[idx].equ_coord.ra_error *= factor;
                    self.observations[idx].equ_coord.dec_error *= factor;
                }
            }
            batch_start = j;
        }

        self
    }
}

#[cfg(test)]
mod test_batch_rms_correction {
    use approx::assert_ulps_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        TrajId,
        coordinates::equatorial::EquCoord,
        observation_dataset::{
            ObsDataset,
            index::{ObsMapIndex, TrajIndexMap},
            observation::ObservationInput,
        },
        observer::{dataset::ObserverId, error_model::ObsErrorModel},
        photometry::{Filter, Photometry},
    };

    fn make_photometry() -> Photometry {
        Photometry {
            magnitude: 15.0,
            error: 0.1,
            filter: Filter::String("V".into()),
        }
    }

    /// Build a minimal `ObservationInput` with the given `id`, observer, and MJD.
    ///
    /// `id` must be unique across observations in the same dataset.
    fn obs(id: u64, observer: Option<ObserverId>, time: f64) -> ObservationInput {
        ObservationInput {
            id,
            equ_coord: EquCoord::new(1.0, 1e-6, 0.5, 2e-6),
            photometry: make_photometry(),
            mjd_tt: time,
            observer,
        }
    }

    /// Wrap a `Vec<ObservationInput>` into an owned `ObsDataset` in which **all**
    /// observations belong to a single trajectory.
    ///
    /// `apply_batch_rms_correction` only forms batches inside a trajectory, so a
    /// trajectory index is required for it to do anything. Placing every
    /// observation in one trajectory isolates the observer/`gap_max` batching
    /// logic, which is what most tests in this module exercise.
    fn dataset(observations: Vec<ObservationInput>) -> ObsDataset {
        let all_indices: Vec<usize> = (0..observations.len()).collect();
        let mut traj_map = TrajIndexMap::new();
        traj_map.insert(TrajId::Int(0), ObsMapIndex::Split(all_indices));
        ObsDataset::new(observations, vec![], None, None, Some(traj_map))
    }

    /// Wrap a `Vec<ObservationInput>` into an `ObsDataset` **without** a
    /// trajectory index, used to exercise the documented no-op fallback.
    fn dataset_no_traj_index(observations: Vec<ObservationInput>) -> ObsDataset {
        ObsDataset::new(observations, vec![], None, None, None)
    }

    /// Wrap a `Vec<ObservationInput>` into an `ObsDataset` whose trajectory index
    /// is built from an explicit `(TrajId, observation positions)` mapping.
    ///
    /// Positions are zero-based indices into `observations`.
    fn dataset_with_trajectories(
        observations: Vec<ObservationInput>,
        trajectories: &[(u32, &[usize])],
    ) -> ObsDataset {
        let mut traj_map = TrajIndexMap::new();
        for (traj_id, indices) in trajectories {
            traj_map.insert(TrajId::Int(*traj_id), ObsMapIndex::Split(indices.to_vec()));
        }
        ObsDataset::new(observations, vec![], None, None, Some(traj_map))
    }

    #[test]
    fn test_single_batch_vfcc17_large() {
        let base_time = 59000.0;
        let observer = Some(ObserverId::MpcCode(*b"A01"));
        let ds = dataset(vec![
            obs(0, observer, base_time),
            obs(1, observer, base_time + 0.01),
            obs(2, observer, base_time + 0.02),
            obs(3, observer, base_time + 0.03),
            obs(4, observer, base_time + 0.04), // n = 5
        ]);

        let corrected = ds
            .with_error_model(ObsErrorModel::VFCC17)
            .apply_batch_rms_correction(8.0 / 24.0);

        let factor = (5.0_f64 * 0.25_f64).sqrt();
        for ob in corrected.iter_observations() {
            assert_ulps_eq!(ob.equ_coord().ra_error, 1e-6 * factor, max_ulps = 2);
            assert_ulps_eq!(ob.equ_coord().dec_error, 2e-6 * factor, max_ulps = 2);
        }
    }

    #[test]
    fn test_single_batch_small_n() {
        let base_time = 59000.0;
        let observer = Some(ObserverId::MpcCode(*b"B01"));
        let ds = dataset(vec![
            obs(0, observer, base_time),
            obs(1, observer, base_time + 0.01), // n = 2
        ]);

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        let factor = (2.0f64).sqrt();
        for ob in corrected.iter_observations() {
            assert_ulps_eq!(ob.equ_coord().ra_error, 1e-6 * factor, max_ulps = 2);
            assert_ulps_eq!(ob.equ_coord().dec_error, 2e-6 * factor, max_ulps = 2);
        }
    }

    #[test]
    fn test_multiple_batches_same_observer() {
        let base_time = 59000.0;
        let observer = Some(ObserverId::MpcCode(*b"C01"));
        let ds = dataset(vec![
            obs(0, observer, base_time),
            obs(1, observer, base_time + 0.01), // batch 1 (n = 2)
            obs(2, observer, base_time + 1.0),  // isolated, batch 2 (n = 1)
        ]);

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        let factor1 = (2.0f64).sqrt();
        let factor2 = 1.0;
        let obs: Vec<_> = corrected.iter_observations().collect();

        assert_ulps_eq!(obs[0].equ_coord().ra_error, 1e-6 * factor1, max_ulps = 2);
        assert_ulps_eq!(obs[1].equ_coord().ra_error, 1e-6 * factor1, max_ulps = 2);
        assert_ulps_eq!(obs[2].equ_coord().ra_error, 1e-6 * factor2, max_ulps = 2);
    }

    #[test]
    fn test_different_observers_are_not_grouped() {
        let base_time = 59000.0;
        let ds = dataset(vec![
            obs(0, Some(ObserverId::MpcCode(*b"D01")), base_time),
            obs(1, Some(ObserverId::MpcCode(*b"D02")), base_time + 0.01),
            obs(2, Some(ObserverId::MpcCode(*b"D03")), base_time + 0.02),
        ]);

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        for ob in corrected.iter_observations() {
            assert_ulps_eq!(ob.equ_coord().ra_error, 1e-6, max_ulps = 2);
            assert_ulps_eq!(ob.equ_coord().dec_error, 2e-6, max_ulps = 2);
        }
    }

    #[test]
    fn test_batch_gaps_exceed_gapmax() {
        let observer = Some(ObserverId::MpcCode(*b"E01"));
        let ds = dataset(vec![
            obs(0, observer, 59000.0),
            obs(1, observer, 59001.0), // > 8h => separate
        ]);

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        for ob in corrected.iter_observations() {
            assert_ulps_eq!(ob.equ_coord().ra_error, 1e-6, max_ulps = 2);
            assert_ulps_eq!(ob.equ_coord().dec_error, 2e-6, max_ulps = 2);
        }
    }

    // ── per-object grouping (regression tests for the batch-size bug) ─────────

    /// Return the `ra_error` of every observation, keyed by observation id.
    fn ra_errors_by_id(ds: &ObsDataset) -> std::collections::HashMap<u64, f64> {
        ds.iter_observations()
            .map(|ob| (*ob.id(), ob.equ_coord().ra_error))
            .collect()
    }

    /// The inflation factor applied to a trajectory must depend only on that
    /// trajectory's own observations, never on how many unrelated trajectories
    /// share the dataset.
    ///
    /// The same object is corrected first on its own, then buried among 200
    /// other trajectories all observed by the same site on the same night. Its
    /// corrected errors must come out identical in both runs.
    #[test]
    fn batch_inflation_does_not_depend_on_dataset_size() {
        let observer = Some(ObserverId::MpcCode(*b"Z01"));
        let base = 59000.0;

        // Target object: 3 observations, one night, one site.
        let target = || {
            vec![
                obs(0, observer, base),
                obs(1, observer, base + 0.01),
                obs(2, observer, base + 0.02),
            ]
        };

        // Run 1: the object alone.
        let small = dataset_with_trajectories(target(), &[(0, &[0, 1, 2])])
            .with_error_model(ObsErrorModel::VFCC17)
            .apply_batch_rms_correction(8.0 / 24.0);

        // Run 2: the same object plus 200 other trajectories, 4 observations
        // each, interleaved in time and all from the same site and night.
        let n_other = 200;
        let mut all = target();
        let mut trajectories: Vec<(u32, Vec<usize>)> = vec![(0, vec![0, 1, 2])];
        let mut next_id = 3u64;
        for t in 0..n_other {
            let mut idx = Vec::new();
            for k in 0..4 {
                idx.push(all.len());
                all.push(obs(
                    next_id,
                    observer,
                    base + 0.001 * (t as f64) + 0.0001 * (k as f64),
                ));
                next_id += 1;
            }
            trajectories.push((t as u32 + 1, idx));
        }
        let traj_refs: Vec<(u32, &[usize])> = trajectories
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        let large = dataset_with_trajectories(all, &traj_refs)
            .with_error_model(ObsErrorModel::VFCC17)
            .apply_batch_rms_correction(8.0 / 24.0);

        let small_err = ra_errors_by_id(&small);
        let large_err = ra_errors_by_id(&large);

        // n = 3 < 5 → factor sqrt(3) in both runs.
        let expected = 1e-6 * (3.0_f64).sqrt();
        for id in [0, 1, 2] {
            assert_ulps_eq!(small_err[&id], expected, max_ulps = 2);
            assert_ulps_eq!(large_err[&id], expected, max_ulps = 2);
            assert_ulps_eq!(small_err[&id], large_err[&id], max_ulps = 2);
        }
    }

    /// For an object observed `n` times from one site within one night, the
    /// factor must be exactly `sqrt(n)` (n < 5) or `sqrt(n / 4)` (n >= 5, under
    /// VFCC17), independent of the rest of the dataset.
    #[test]
    fn factor_equals_sqrt_of_same_object_count() {
        let observer = Some(ObserverId::MpcCode(*b"Z02"));
        let base = 59000.0;

        for n in 1usize..=8 {
            // The object under test.
            let mut observations: Vec<ObservationInput> = (0..n)
                .map(|k| obs(k as u64, observer, base + 0.005 * k as f64))
                .collect();
            let target_idx: Vec<usize> = (0..n).collect();

            // A second, unrelated object with many observations the same night.
            let other_start = observations.len();
            for k in 0..20 {
                observations.push(obs(1000 + k as u64, observer, base + 0.005 * k as f64));
            }
            let other_idx: Vec<usize> = (other_start..observations.len()).collect();

            let corrected =
                dataset_with_trajectories(observations, &[(0, &target_idx), (1, &other_idx)])
                    .with_error_model(ObsErrorModel::VFCC17)
                    .apply_batch_rms_correction(8.0 / 24.0);

            let expected_factor = if n >= 5 {
                (n as f64 * 0.25).sqrt()
            } else {
                (n as f64).sqrt()
            };
            let err = ra_errors_by_id(&corrected);
            for k in 0..n as u64 {
                assert_ulps_eq!(err[&k], 1e-6 * expected_factor, max_ulps = 2);
            }
        }
    }

    /// Two observations of the *same* object from the same site, but far apart
    /// in time, must not share a batch: the `gap_max` split still applies inside
    /// a trajectory.
    #[test]
    fn gap_split_applies_within_a_trajectory() {
        let observer = Some(ObserverId::MpcCode(*b"Z03"));
        let ds = dataset_with_trajectories(
            vec![
                obs(0, observer, 59000.0),
                obs(1, observer, 59000.1), // same night → batch with id 0
                obs(2, observer, 59002.0), // > gap_max later → its own batch
            ],
            &[(0, &[0, 1, 2])],
        );

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        let err = ra_errors_by_id(&corrected);
        assert_ulps_eq!(err[&0], 1e-6 * (2.0_f64).sqrt(), max_ulps = 2);
        assert_ulps_eq!(err[&1], 1e-6 * (2.0_f64).sqrt(), max_ulps = 2);
        assert_ulps_eq!(err[&2], 1e-6, max_ulps = 2);
    }

    /// Observations of one object taken from two different sites in the same
    /// night are batched per site, not merged.
    #[test]
    fn distinct_observers_within_a_trajectory_are_not_merged() {
        let obs_a = Some(ObserverId::MpcCode(*b"Z04"));
        let obs_b = Some(ObserverId::MpcCode(*b"Z05"));
        let ds = dataset_with_trajectories(
            vec![
                obs(0, obs_a, 59000.00),
                obs(1, obs_a, 59000.01),
                obs(2, obs_a, 59000.02), // site A: batch of 3
                obs(3, obs_b, 59000.03),
                obs(4, obs_b, 59000.04), // site B: batch of 2
            ],
            &[(0, &[0, 1, 2, 3, 4])],
        );

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        let err = ra_errors_by_id(&corrected);
        for id in [0, 1, 2] {
            assert_ulps_eq!(err[&id], 1e-6 * (3.0_f64).sqrt(), max_ulps = 2);
        }
        for id in [3, 4] {
            assert_ulps_eq!(err[&id], 1e-6 * (2.0_f64).sqrt(), max_ulps = 2);
        }
    }

    /// An observation listed under two trajectories is corrected exactly once
    /// (single multiplication by the factor of one batch, not both).
    #[test]
    fn observation_in_two_trajectories_is_corrected_once() {
        let observer = Some(ObserverId::MpcCode(*b"Z06"));
        // Observation 2 is shared between trajectory 0 and trajectory 1.
        let ds = dataset_with_trajectories(
            vec![
                obs(0, observer, 59000.00),
                obs(1, observer, 59000.01),
                obs(2, observer, 59000.02), // shared
                obs(3, observer, 59000.03),
            ],
            &[(0, &[0, 1, 2]), (1, &[2, 3])],
        );

        let corrected = ds
            .with_error_model(ObsErrorModel::FCCT14)
            .apply_batch_rms_correction(8.0 / 24.0);

        let err = ra_errors_by_id(&corrected);
        // Whichever single group obs 2 lands in, its error is 1e-6 times the
        // sqrt of that group's size (2 or 3): never the product of both.
        let f2 = (2.0_f64).sqrt();
        let f3 = (3.0_f64).sqrt();
        assert!(
            (err[&2] - 1e-6 * f2).abs() < 1e-14 || (err[&2] - 1e-6 * f3).abs() < 1e-14,
            "obs 2 corrected more than once: {}",
            err[&2]
        );
    }

    /// Without a trajectory index the correction cannot identify same-object
    /// batches and must degrade to a no-op, even for many same-site
    /// observations packed into one night.
    #[test]
    fn no_trajectory_index_is_a_noop() {
        let observer = Some(ObserverId::MpcCode(*b"Z07"));
        let observations: Vec<ObservationInput> = (0..50)
            .map(|k| obs(k, observer, 59000.0 + 0.001 * k as f64))
            .collect();

        let corrected = dataset_no_traj_index(observations)
            .with_error_model(ObsErrorModel::VFCC17)
            .apply_batch_rms_correction(8.0 / 24.0);

        for ob in corrected.iter_observations() {
            assert_ulps_eq!(ob.equ_coord().ra_error, 1e-6, max_ulps = 2);
            assert_ulps_eq!(ob.equ_coord().dec_error, 2e-6, max_ulps = 2);
        }
    }

    // ── proptest helpers ─────────────────────────────────────────────────────

    /// Build an `Observation` with explicit coordinate errors from proptest inputs.
    fn obs_with_errors(
        id: u64,
        observer: Option<ObserverId>,
        time: f64,
        ra: f64,
        ra_error: f64,
        dec: f64,
        dec_error: f64,
    ) -> ObservationInput {
        ObservationInput {
            id,
            equ_coord: EquCoord::new(ra, ra_error, dec, dec_error),
            photometry: make_photometry(),
            mjd_tt: time,
            observer,
        }
    }

    // ── proptest: errors never decrease after batch correction ────────────────

    proptest! {
        /// For any batch of observations from the same observer within gap_max,
        /// every `ra_error` and `dec_error` after correction must be ≥ the original.
        /// The batch correction factor is always `sqrt(n) >= 1` (or `sqrt(n*0.25)` for
        /// VFCC17 with n≥5, which is ≥ 1 when n≥4; but n≥5 guarantees factor≥√1.25>1).
        #[test]
        fn prop_errors_never_decrease(
            ra_errors in prop::collection::vec(1e-9..1e-3f64, 1..=20usize),
            dec_errors in prop::collection::vec(1e-9..1e-3f64, 1..=20usize),
            base_time in 59000.0..60000.0f64,
        ) {
            // Use the shorter of the two vecs so they zip cleanly
            let n = ra_errors.len().min(dec_errors.len());
            let observer = Some(ObserverId::MpcCode(*b"F01"));
            // Space observations 0.01 days apart — well within the 8h gap_max
            let observations: Vec<ObservationInput> = (0..n)
                .map(|i| obs_with_errors(
                    i as u64,
                    observer,
                    base_time + i as f64 * 0.01,
                    0.5,
                    ra_errors[i],
                    0.3,
                    dec_errors[i],
                ))
                .collect();

            let original_ra: Vec<f64> = observations.iter()
                .map(|o| o.equ_coord.ra_error)
                .collect();
            let original_dec: Vec<f64> = observations.iter()
                .map(|o| o.equ_coord.dec_error)
                .collect();

            let corrected = dataset(observations)
                .with_error_model(ObsErrorModel::FCCT14)
                .apply_batch_rms_correction(8.0 / 24.0);

            let corrected_obs: Vec<_> = corrected.iter_observations().collect();
            prop_assert_eq!(corrected_obs.len(), n);
            for (ob, (&orig_ra, &orig_dec)) in
                corrected_obs.iter().zip(original_ra.iter().zip(original_dec.iter()))
            {
                prop_assert!(
                    ob.equ_coord().ra_error >= orig_ra - f64::EPSILON,
                    "ra_error decreased: {} < {}",
                    ob.equ_coord().ra_error,
                    orig_ra
                );
                prop_assert!(
                    ob.equ_coord().dec_error >= orig_dec - f64::EPSILON,
                    "dec_error decreased: {} < {}",
                    ob.equ_coord().dec_error,
                    orig_dec
                );
            }
        }
    }

    // ── proptest: single observation → factor 1, errors unchanged ────────────

    proptest! {
        /// A dataset with exactly one observation must have its errors unchanged after
        /// correction, because the batch has size 1 and sqrt(1) = 1.
        #[test]
        fn prop_single_observation_errors_unchanged(
            ra in -1.5..1.5f64,
            dec in -1.5..1.5f64,
            ra_error in 1e-9..1e-3f64,
            dec_error in 1e-9..1e-3f64,
            time in 59000.0..60000.0f64,
        ) {
            let observer = Some(ObserverId::MpcCode(*b"G01"));
            let observation = obs_with_errors(0, observer, time, ra, ra_error, dec, dec_error);
            let ds = dataset(vec![observation]);

            let corrected = ds.with_error_model(ObsErrorModel::FCCT14).apply_batch_rms_correction(8.0 / 24.0);

            let obs: Vec<_> = corrected.iter_observations().collect();
            prop_assert_eq!(obs.len(), 1);
            // Factor must be sqrt(1) = 1, so errors are unchanged.
            prop_assert!(
                (obs[0].equ_coord().ra_error - ra_error).abs() < f64::EPSILON * ra_error,
                "ra_error changed for single-obs batch: {} vs {}",
                obs[0].equ_coord().ra_error,
                ra_error
            );
            prop_assert!(
                (obs[0].equ_coord().dec_error - dec_error).abs() < f64::EPSILON * dec_error,
                "dec_error changed for single-obs batch: {} vs {}",
                obs[0].equ_coord().dec_error,
                dec_error
            );
        }
    }

    // ── proptest: all different observers → every batch size 1 → unchanged ───

    proptest! {
        /// When every observation comes from a distinct observer, each forms its own
        /// batch of size 1 and errors must be unchanged (factor = sqrt(1) = 1).
        #[test]
        fn prop_all_different_observers_errors_unchanged(
            ra_errors in prop::collection::vec(1e-9..1e-3f64, 1..=10usize),
            dec_errors in prop::collection::vec(1e-9..1e-3f64, 1..=10usize),
            base_time in 59000.0..60000.0f64,
        ) {
            let n = ra_errors.len().min(dec_errors.len());
            // Give every observation a unique MPC code derived from its index.
            let observations: Vec<ObservationInput> = (0..n)
                .map(|i| {
                    // Build a 3-byte code that encodes the index uniquely.
                    let b0 = b'A' + (i / 26) as u8;
                    let b1 = b'A' + (i % 26) as u8;
                    let observer = Some(ObserverId::MpcCode([b0, b1, b'0']));
                    obs_with_errors(
                        i as u64,
                        observer,
                        base_time + i as f64 * 0.01,
                        0.5,
                        ra_errors[i],
                        0.3,
                        dec_errors[i],
                    )
                })
                .collect();

            let original_ra: Vec<f64> = observations.iter()
                .map(|o| o.equ_coord.ra_error)
                .collect();
            let original_dec: Vec<f64> = observations.iter()
                .map(|o| o.equ_coord.dec_error)
                .collect();

            let corrected = dataset(observations)
                .with_error_model(ObsErrorModel::FCCT14)
                .apply_batch_rms_correction(8.0 / 24.0);

            let corrected_obs: Vec<_> = corrected.iter_observations().collect();
            prop_assert_eq!(corrected_obs.len(), n);
            for (ob, (&orig_ra, &orig_dec)) in
                corrected_obs.iter().zip(original_ra.iter().zip(original_dec.iter()))
            {
                prop_assert!(
                    (ob.equ_coord().ra_error - orig_ra).abs() < f64::EPSILON * orig_ra,
                    "ra_error changed for distinct-observer batch: {} vs {}",
                    ob.equ_coord().ra_error,
                    orig_ra
                );
                prop_assert!(
                    (ob.equ_coord().dec_error - orig_dec).abs() < f64::EPSILON * orig_dec,
                    "dec_error changed for distinct-observer batch: {} vs {}",
                    ob.equ_coord().dec_error,
                    orig_dec
                );
            }
        }
    }

    // ── proptest: VFCC17 with n < 5 uses sqrt(n), same as FCCT14 ─────────────

    proptest! {
        /// For a batch of size 1..=4, VFCC17 must produce the exact same factor as
        /// FCCT14 (both use `sqrt(n)`), so the resulting errors are identical.
        #[test]
        fn prop_vfcc17_small_batch_same_as_fcct14(
            ra_error in 1e-9..1e-3f64,
            dec_error in 1e-9..1e-3f64,
            base_time in 59000.0..60000.0f64,
            // batch size in [1, 4]: VFCC17 special branch requires n >= 5
            extra in 0usize..4usize,
        ) {
            let observer = Some(ObserverId::MpcCode(*b"H01"));
            let n = extra + 1; // 1..=4
            let make_obs = || {
                (0..n)
                    .map(|i| obs_with_errors(
                        i as u64,
                        observer,
                        base_time + i as f64 * 0.01,
                        0.5,
                        ra_error,
                        0.3,
                        dec_error,
                    ))
                    .collect::<Vec<_>>()
            };

            let corrected_vfcc17 = dataset(make_obs())
                .with_error_model(ObsErrorModel::VFCC17)
                .apply_batch_rms_correction(8.0 / 24.0);
            let corrected_fcct14 = dataset(make_obs())
                .with_error_model(ObsErrorModel::FCCT14)
                .apply_batch_rms_correction(8.0 / 24.0);

            let vfcc17_obs: Vec<_> = corrected_vfcc17.iter_observations().collect();
            let fcct14_obs: Vec<_> = corrected_fcct14.iter_observations().collect();

            prop_assert_eq!(vfcc17_obs.len(), fcct14_obs.len());
            for (v, f) in vfcc17_obs.iter().zip(fcct14_obs.iter()) {
                prop_assert!(
                    (v.equ_coord().ra_error - f.equ_coord().ra_error).abs()
                        < f64::EPSILON * f.equ_coord().ra_error,
                    "VFCC17 and FCCT14 ra_error differ for n={}: {} vs {}",
                    n,
                    v.equ_coord().ra_error,
                    f.equ_coord().ra_error
                );
                prop_assert!(
                    (v.equ_coord().dec_error - f.equ_coord().dec_error).abs()
                        < f64::EPSILON * f.equ_coord().dec_error,
                    "VFCC17 and FCCT14 dec_error differ for n={}: {} vs {}",
                    n,
                    v.equ_coord().dec_error,
                    f.equ_coord().dec_error
                );
            }
        }
    }

    // ── proptest: no error model → apply_model_errors is a no-op ─────────────

    proptest! {
        /// When no error model is attached to the dataset, `apply_model_errors` must
        /// return the dataset with every `ra_error` and `dec_error` unchanged.
        #[test]
        fn prop_no_error_model_apply_model_errors_is_noop(
            ra in -1.5..1.5f64,
            dec in -1.5..1.5f64,
            ra_error in 1e-9..1e-3f64,
            dec_error in 1e-9..1e-3f64,
            time in 59000.0..60000.0f64,
        ) {
            let observer = Some(ObserverId::MpcCode(*b"I01"));
            let observation = obs_with_errors(0, observer, time, ra, ra_error, dec, dec_error);
            // Build dataset with NO error model (None for the model argument).
            let ds = ObsDataset::new(vec![observation], vec![], None, None, None);

            let result = ds.apply_model_errors();

            let obs: Vec<_> = result.iter_observations().collect();
            prop_assert_eq!(obs.len(), 1);
            prop_assert!(
                (obs[0].equ_coord().ra_error - ra_error).abs() < f64::EPSILON * ra_error,
                "ra_error changed without error model: {} vs {}",
                obs[0].equ_coord().ra_error,
                ra_error
            );
            prop_assert!(
                (obs[0].equ_coord().dec_error - dec_error).abs() < f64::EPSILON * dec_error,
                "dec_error changed without error model: {} vs {}",
                obs[0].equ_coord().dec_error,
                dec_error
            );
        }
    }

    mod multiple_observer_case {
        use super::*;

        // ── interleaved observers ─────────────────────────────────────────────────

        /// Two observers whose observations are interleaved in time.
        ///
        /// Timeline (days):
        /// ```text
        /// t=0.00  obs A (obs_A1)
        /// t=0.01  obs B (obs_B1)
        /// t=0.02  obs A (obs_A2)
        /// t=0.03  obs B (obs_B2)
        /// t=0.04  obs A (obs_A3)
        /// t=0.05  obs B (obs_B3)
        /// ```
        ///
        /// Each observer forms a single batch of 3 observations.
        /// Expected factor: `sqrt(3)` for both (FCCT14).
        #[test]
        fn test_two_observers_interleaved_single_batch_each() {
            let obs_a = Some(ObserverId::MpcCode(*b"J01"));
            let obs_b = Some(ObserverId::MpcCode(*b"J02"));

            let ds = dataset(vec![
                obs(0, obs_a, 59000.00),
                obs(1, obs_b, 59000.01),
                obs(2, obs_a, 59000.02),
                obs(3, obs_b, 59000.03),
                obs(4, obs_a, 59000.04),
                obs(5, obs_b, 59000.05),
            ]);

            let corrected = ds
                .with_error_model(ObsErrorModel::FCCT14)
                .apply_batch_rms_correction(8.0 / 24.0);

            let factor = (3.0f64).sqrt();
            for ob in corrected.iter_observations() {
                assert_ulps_eq!(ob.equ_coord().ra_error, 1e-6 * factor, max_ulps = 2);
                assert_ulps_eq!(ob.equ_coord().dec_error, 2e-6 * factor, max_ulps = 2);
            }
        }

        /// Two observers interleaved in time, but observer B has a gap that splits
        /// its observations into two batches. Observer A's last observation also
        /// falls after the gap.
        ///
        /// Timeline (days):
        /// ```text
        /// t=0.00  obs A (id=0)  ─┐ batch A1 (n=3)
        /// t=0.01  obs B (id=1)   │ batch B1 (n=2)
        /// t=0.02  obs A (id=2)   │
        /// t=0.03  obs B (id=3)  ─┘
        /// t=0.04  obs A (id=4)  ─┘
        /// t=1.00  obs B (id=5)  ── batch B2 (n=1), gap > 8h from B batch 1
        /// t=1.01  obs A (id=6)  ── batch A2 (n=1), gap > 8h from A batch 1
        /// ```
        ///
        /// Observer A batch 1 (ids 0,2,4): factor `sqrt(3)`.
        /// Observer A batch 2 (id 6):      factor 1.
        /// Observer B batch 1 (ids 1,3):   factor `sqrt(2)`.
        /// Observer B batch 2 (id 5):      factor 1.
        #[test]
        fn test_two_observers_interleaved_one_has_gap() {
            let obs_a = Some(ObserverId::MpcCode(*b"K01"));
            let obs_b = Some(ObserverId::MpcCode(*b"K02"));

            let ds = dataset(vec![
                obs(0, obs_a, 59000.00),
                obs(1, obs_b, 59000.01),
                obs(2, obs_a, 59000.02),
                obs(3, obs_b, 59000.03),
                obs(4, obs_a, 59000.04),
                obs(5, obs_b, 59001.00), // gap > 8h from obs B batch 1
                obs(6, obs_a, 59001.01), // gap > 8h from obs A batch 1
            ]);

            let corrected = ds
                .with_error_model(ObsErrorModel::FCCT14)
                .apply_batch_rms_correction(8.0 / 24.0);

            let mut by_id: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
            for ob in corrected.iter_observations() {
                by_id.insert(*ob.id(), ob.equ_coord().ra_error);
            }

            let factor_a1 = (3.0f64).sqrt(); // ids 0, 2, 4
            let factor_a2 = 1.0f64; // id 6
            let factor_b1 = (2.0f64).sqrt(); // ids 1, 3
            let factor_b2 = 1.0f64; // id 5

            assert_ulps_eq!(by_id[&0], 1e-6 * factor_a1, max_ulps = 2);
            assert_ulps_eq!(by_id[&2], 1e-6 * factor_a1, max_ulps = 2);
            assert_ulps_eq!(by_id[&4], 1e-6 * factor_a1, max_ulps = 2);
            assert_ulps_eq!(by_id[&6], 1e-6 * factor_a2, max_ulps = 2);

            assert_ulps_eq!(by_id[&1], 1e-6 * factor_b1, max_ulps = 2);
            assert_ulps_eq!(by_id[&3], 1e-6 * factor_b1, max_ulps = 2);
            assert_ulps_eq!(by_id[&5], 1e-6 * factor_b2, max_ulps = 2);
        }

        /// Three observers interleaved, each forming multiple batches, with VFCC17
        /// triggered for the large batch.
        ///
        /// Timeline (days):
        /// ```text
        /// t=0.00  obs A (obs_A1)  ─┐
        /// t=0.01  obs B (obs_B1)   │
        /// t=0.02  obs C (obs_C1)   │ all within 8h
        /// t=0.03  obs A (obs_A2)   │
        /// t=0.04  obs B (obs_B2)   │
        /// t=0.05  obs C (obs_C2)   │
        /// t=0.06  obs A (obs_A3)   │
        /// t=0.07  obs B (obs_B3)   │
        /// t=0.08  obs C (obs_C3)   │
        /// t=0.09  obs A (obs_A4)   │
        /// t=0.10  obs B (obs_B4)   │
        /// t=0.11  obs A (obs_A5)  ─┘  A: n=5, B: n=4, C: n=3
        ///
        /// t=2.00  obs C (obs_C4)  ── batch C2: n=1 (isolated)
        /// ```
        ///
        /// Under VFCC17:
        /// - Observer A (n=5): factor = `sqrt(5 * 0.25)` = `sqrt(1.25)`
        /// - Observer B (n=4): factor = `sqrt(4)` = 2  (n < 5, fallback to sqrt(n))
        /// - Observer C batch 1 (n=3): factor = `sqrt(3)`
        /// - Observer C batch 2 (n=1): factor = 1
        #[test]
        fn test_three_observers_interleaved_vfcc17() {
            let obs_a = Some(ObserverId::MpcCode(*b"L01"));
            let obs_b = Some(ObserverId::MpcCode(*b"L02"));
            let obs_c = Some(ObserverId::MpcCode(*b"L03"));

            let ds = dataset(vec![
                obs(0, obs_a, 59000.00),
                obs(1, obs_b, 59000.01),
                obs(2, obs_c, 59000.02),
                obs(3, obs_a, 59000.03),
                obs(4, obs_b, 59000.04),
                obs(5, obs_c, 59000.05),
                obs(6, obs_a, 59000.06),
                obs(7, obs_b, 59000.07),
                obs(8, obs_c, 59000.08),
                obs(9, obs_a, 59000.09),
                obs(10, obs_b, 59000.10),
                obs(11, obs_a, 59000.11),
                obs(12, obs_c, 59002.00), // isolated batch for C
            ]);

            let corrected = ds
                .with_error_model(ObsErrorModel::VFCC17)
                .apply_batch_rms_correction(8.0 / 24.0);

            let mut by_id: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
            for ob in corrected.iter_observations() {
                by_id.insert(*ob.id(), ob.equ_coord().ra_error);
            }

            let factor_a = (5.0_f64 * 0.25).sqrt(); // n=5, VFCC17 branch
            let factor_b = (4.0_f64).sqrt(); // n=4, fallback
            let factor_c1 = (3.0_f64).sqrt(); // n=3, fallback
            let factor_c2 = 1.0_f64; // n=1

            for id in [0, 3, 6, 9, 11] {
                assert_ulps_eq!(by_id[&id], 1e-6 * factor_a, max_ulps = 2);
            }
            for id in [1, 4, 7, 10] {
                assert_ulps_eq!(by_id[&id], 1e-6 * factor_b, max_ulps = 2);
            }
            for id in [2, 5, 8] {
                assert_ulps_eq!(by_id[&id], 1e-6 * factor_c1, max_ulps = 2);
            }
            assert_ulps_eq!(by_id[&12], 1e-6 * factor_c2, max_ulps = 2);
        }

        // ── proptest: interleaved observers never contaminate each other ──────────

        proptest! {
            /// For any two observers whose observations are strictly interleaved in time,
            /// each observer's batch factor must match what would be computed if the
            /// observations of that observer were alone in the dataset.
            ///
            /// This verifies that temporal interleaving between observers does not cause
            /// cross-contamination of batch membership.
            #[test]
            fn prop_interleaved_observers_independent_factors(
                n_a in 1usize..=10usize,
                n_b in 1usize..=10usize,
                base_time in 59000.0..60000.0f64,
            ) {
                let obs_a = Some(ObserverId::MpcCode(*b"M01"));
                let obs_b = Some(ObserverId::MpcCode(*b"M02"));

                // Interleave: A at even slots, B at odd slots, 0.01-day spacing.
                let total = n_a + n_b;
                let mut observations = Vec::with_capacity(total);
                let mut id_a = vec![];
                let mut id_b = vec![];
                let mut ia = 0usize;
                let mut ib = 0usize;
                let mut slot = 0usize;
                let mut id = 0u64;

                while ia < n_a || ib < n_b {
                    let time = base_time + slot as f64 * 0.01;
                    if ia < n_a && (slot.is_multiple_of(2) || ib >= n_b) {
                        observations.push(obs(id, obs_a, time));
                        id_a.push(id);
                        ia += 1;
                    } else {
                        observations.push(obs(id, obs_b, time));
                        id_b.push(id);
                        ib += 1;
                    }
                    id += 1;
                    slot += 1;
                }

                // Expected factors from isolated runs.
                let expected_a = (n_a as f64).sqrt();
                let expected_b = (n_b as f64).sqrt();

                let corrected = dataset(observations)
                    .with_error_model(ObsErrorModel::FCCT14)
                    .apply_batch_rms_correction(8.0 / 24.0);

                let mut by_id: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
                for ob in corrected.iter_observations() {
                    by_id.insert(*ob.id(), ob.equ_coord().ra_error);
                }

                for aid in &id_a {
                    prop_assert!(
                        (by_id[aid] - 1e-6 * expected_a).abs() < 1e-12,
                        "observer A contaminated: got {}, expected {}",
                        by_id[aid], 1e-6 * expected_a
                    );
                }
                for bid in &id_b {
                    prop_assert!(
                        (by_id[bid] - 1e-6 * expected_b).abs() < 1e-12,
                        "observer B contaminated: got {}, expected {}",
                        by_id[bid], 1e-6 * expected_b
                    );
                }
            }
        }
    }
}
