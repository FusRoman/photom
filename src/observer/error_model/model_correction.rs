use crate::{
    constants::ARCSEC_TO_RAD,
    observation_dataset::ObsDataset,
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
    /// Observations from the same observer that are closely spaced in time are
    /// grouped into batches. A scaling factor derived from the batch size is then
    /// applied to the `ra_error` and `dec_error` of every observation in the batch.
    ///
    /// ### Grouping
    /// 
    /// Two observations belong to the same batch if and only if:
    ///
    /// - they share the same `observer` identity, **and**
    /// - the time gap between consecutive observations (sorted by MJD) is strictly
    ///   less than `gap_max`.
    ///
    /// Observations from different observers are **always** grouped independently,
    /// even when their timestamps interleave in time.
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
    /// Both `ra_error` and `dec_error` are multiplied by the same factor:
    ///
    /// $$\sigma' = \sigma \times \sqrt{n}$$
    ///
    /// # Arguments
    /// 
    /// - `gap_max` – Maximum time gap (days) between two consecutive observations
    ///   of the same observer for them to be considered part of the same batch.
    ///   A typical value is $8/24 \approx 0.333$ days (8 hours).
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
            Some(ref em) => em,
            None => return self,
        };

        let n_obs = self.observations.len();
        if n_obs == 0 {
            return self;
        }

        // Single allocation: sort indices by (observer, mjd_tt).
        let mut sorted_indices: Vec<usize> = (0..n_obs).collect();
        sorted_indices.sort_unstable_by(|&a, &b| {
            let oa = &self.observations[a];
            let ob = &self.observations[b];
            oa.observer
                .cmp(&ob.observer)
                .then_with(|| oa.mjd_tt.partial_cmp(&ob.mjd_tt).unwrap())
        });

        let mut i = 0;
        while i < n_obs {
            let current_observer = self.observations[sorted_indices[i]].observer;
            let mut batch_start = i;
            let mut j = i + 1;

            // Walk all observations belonging to current_observer.
            loop {
                let end_of_observer =
                    j == n_obs || self.observations[sorted_indices[j]].observer != current_observer;

                let end_of_batch = !end_of_observer && {
                    let prev_time = self.observations[sorted_indices[j - 1]].mjd_tt;
                    let curr_time = self.observations[sorted_indices[j]].mjd_tt;
                    (curr_time - prev_time) > gap_max
                };

                if end_of_observer || end_of_batch {
                    // Flush batch [batch_start, j).
                    let n = j - batch_start;
                    let factor = match error_model {
                        ObsErrorModel::VFCC17 if n >= 5 => (n as f64 * 0.25).sqrt(),
                        _ => (n as f64).sqrt(),
                    };
                    for k in batch_start..j {
                        let idx = sorted_indices[k];
                        self.observations[idx].equ_coord.ra_error *= factor;
                        self.observations[idx].equ_coord.dec_error *= factor;
                    }
                    batch_start = j;
                }

                if end_of_observer {
                    break;
                }

                j += 1;
            }

            i = j;
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
        coordinates::equatorial::EquCoord,
        observation_dataset::{ObsDataset, observation::ObservationInput},
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

    /// Wrap a `Vec<ObservationInput>` into an owned `ObsDataset` (no error model, no index).
    fn dataset(observations: Vec<ObservationInput>) -> ObsDataset {
        ObsDataset::new(observations, vec![], None, None, None)
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
                    if ia < n_a && (slot % 2 == 0 || ib >= n_b) {
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
