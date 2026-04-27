use crate::{
    constants::ARCSEC_TO_DEG,
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

    /// Apply RMS correction based on temporally clustered batches of observations.
    ///
    /// This method adjusts the astrometric uncertainties (`error_ra`, `error_dec`) of each observation
    /// based on the local density of observations in time and observer identity. Observations that are
    /// close in time (within 8 hours) and come from the same observer are grouped into batches, and a
    /// correction factor is applied to reflect statistical correlation or improvement due to redundancy.
    ///
    /// # Warning
    ///
    /// If no error model is attached to the dataset, this method will return `self` unmodified without applying any correction.
    /// Call `with_error_model` or `set_error_model` to attach an error model before invoking this method.
    ///
    /// # Behavior
    ///
    /// - Observations are grouped by `observer` and sorted in time.
    /// - A batch is formed when consecutive observations from the same observer are spaced by less than 8 hours.
    /// - Each observation in a batch of size `n` receives a correction:
    ///     - `√n` for standard models,
    ///     - `√(n × 0.25)` for `vfcc17` when `n ≥ 5`.
    /// - If `n < 5` with `vfcc17`, it falls back to `√n`.
    /// - Observations with fixed weights (`force_w`) are not affected (not yet implemented in this version).
    ///
    /// # Arguments
    ///
    /// - `error_model` - The error model to use when applying the batch correction. Supported values include:
    ///     - `"vfcc17"`: uses a reduced factor `√(n × 0.25)` if the batch has at least 5 observations,
    ///     - any other string: uses the standard `√n` factor.
    /// - `gap_max` - The maximum time gap (in days) to consider observations as part of the same batch.
    ///
    /// # Returns
    ///
    /// - `()` - This function modifies the observations in-place; it does not return a value.
    ///
    /// # Computation Details
    /// ----------
    /// - The time comparison is based on Modified Julian Date (`MJD`), and the batch window is fixed at 8 hours (i.e., `8.0 / 24.0` days).
    /// - The error fields `error_ra` and `error_dec` are both scaled by the same batch correction factor.
    ///
    /// # Units
    /// ----------
    /// - Input and output uncertainties (`error_ra`, `error_dec`) are expressed in **radians**.
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

        // ARCSEC_TO_DEG = 1/3600; multiplying by π/180 gives π/648000 rad/arcsec.
        let arcsec_to_rad = ARCSEC_TO_DEG * std::f64::consts::PI / 180.0;

        for obs in &mut self.observations {
            let mpc_code = match obs.observer {
                Some(ObserverId::MpcCode(code)) => code,
                _ => continue,
            };

            if let Some((rms_ra, rms_dec)) = get_bias_rms(&model_data, mpc_code, "c") {
                let cos_dec = obs.equ_coord.dec.cos();

                let model_ra_rad = rms_ra as f64 * arcsec_to_rad / cos_dec;
                let model_dec_rad = rms_dec as f64 * arcsec_to_rad;

                obs.equ_coord.ra_error = obs.equ_coord.ra_error.max(model_ra_rad);
                obs.equ_coord.dec_error = obs.equ_coord.dec_error.max(model_dec_rad);
            }
        }

        self
    }

    fn apply_batch_rms_correction(mut self, gap_max: f64) -> ObsDataset {
        let error_model = match self.observer_dataset.mpc_error_model {
            Some(ref em) => em,
            None => return self,
        };

        self.observations
            .sort_by(|a, b| a.mjd_tt.partial_cmp(&b.mjd_tt).unwrap());

        let n_obs = self.observations.len();

        if n_obs == 0 {
            return self;
        }

        let mut i = 0;
        while i < n_obs {
            let observer = self.observations[i].observer;
            let mut batch_indices: Vec<usize> = vec![i];
            let mut j = i + 1;

            while j < n_obs {
                let obs_j = &self.observations[j];
                let prev_time = self.observations[*batch_indices.last().unwrap()].mjd_tt;

                if obs_j.observer == observer && (obs_j.mjd_tt - prev_time) <= gap_max {
                    batch_indices.push(j);
                    j += 1;
                } else {
                    break;
                }
            }

            let n = batch_indices.len();
            let factor = match error_model {
                ObsErrorModel::VFCC17 if n >= 5 => (n as f64 * 0.25).sqrt(),
                _ => (n as f64).sqrt(),
            };

            for idx in &batch_indices {
                self.observations[*idx].equ_coord.ra_error *= factor;
                self.observations[*idx].equ_coord.dec_error *= factor;
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
        observation_dataset::{ObsDataset, observation::Observation},
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

    /// Build a minimal `Observation` with the given `id`, observer, and MJD.
    ///
    /// `id` must be unique across observations in the same dataset.
    fn obs(id: u64, observer: Option<ObserverId>, time: f64) -> Observation {
        Observation {
            index: None,
            id,
            equ_coord: EquCoord::new(1.0, 1e-6, 0.5, 2e-6),
            photometry: make_photometry(),
            mjd_tt: time,
            observer,
        }
    }

    /// Wrap a `Vec<Observation>` into an owned `ObsDataset` (no error model, no index).
    fn dataset(observations: Vec<Observation>) -> ObsDataset {
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
    ) -> Observation {
        Observation {
            index: None,
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
            let observations: Vec<Observation> = (0..n)
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
            let observations: Vec<Observation> = (0..n)
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
}
