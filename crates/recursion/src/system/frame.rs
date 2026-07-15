use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::{
        MultiStarkVerifyingKey, StarkVerifyingKey, StarkVerifyingParams,
        VerifierSinglePreprocessedData,
    },
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest, F};

use crate::whir::whir_round_encoder;

/*
 * Modified versions of the STARK and multi-STARK verifying keys for AirModule
 * implementations. AirModules should use MultiStarkVerifyingKeyFrame instead
 * of MultiStarkVerifyingKey<BabyBearPoseidon2Config> in their AIRs, as use of
 * some fields in the latter will compromise internal vk stability.
 *
 * We also define check_param_compatibility, which asserts compatibility (with
 * regards to vk stability) between given app, leaf, and internal SystemParams.
 *
 * For more information on vk stability and what can be used in AIRs and how,
 * see crates/recursion/README.md.
 */

#[derive(Clone)]
pub struct StarkVkeyFrame {
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
    pub params: StarkVerifyingParams,
    pub num_interactions: usize,
    pub max_constraint_degree: u8,
    pub is_required: bool,
}

#[derive(Clone)]
pub struct MultiStarkVkeyFrame {
    pub params: SystemParams,
    pub per_air: Vec<StarkVkeyFrame>,
    pub max_constraint_degree: usize,
}

impl From<&StarkVerifyingKey<F, Digest>> for StarkVkeyFrame {
    fn from(vk: &StarkVerifyingKey<F, Digest>) -> Self {
        Self {
            preprocessed_data: vk.preprocessed_data.clone(),
            params: vk.params.clone(),
            num_interactions: vk.num_interactions(),
            max_constraint_degree: vk.max_constraint_degree,
            is_required: vk.is_required,
        }
    }
}

impl From<&MultiStarkVerifyingKey<BabyBearPoseidon2Config>> for MultiStarkVkeyFrame {
    fn from(mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>) -> Self {
        Self {
            params: mvk.inner.params.clone(),
            per_air: mvk.inner.per_air.iter().map(Into::into).collect_vec(),
            max_constraint_degree: mvk.max_constraint_degree(),
        }
    }
}

pub fn check_param_compatibility(
    app_params: &SystemParams,
    leaf_params: &SystemParams,
    internal_params: &SystemParams,
) {
    // num_whir_rounds affects the number of columns in WhirRoundAir.
    assert_eq!(
        whir_round_encoder(leaf_params.num_whir_rounds()).width(),
        whir_round_encoder(internal_params.num_whir_rounds()).width()
    );
    // logup_pow_bits affects the number of interactions in GkrInputAir.
    assert_eq!(
        app_params.logup_pow_bits() > 0,
        leaf_params.logup_pow_bits() > 0
    );
    assert_eq!(
        leaf_params.logup_pow_bits() > 0,
        internal_params.logup_pow_bits() > 0
    );
    // mu_pow_bits affects the number of interactions in StackingClaimsAir.
    assert_eq!(
        app_params.whir.mu_pow_bits > 0,
        leaf_params.whir.mu_pow_bits > 0
    );
    assert_eq!(
        leaf_params.whir.mu_pow_bits > 0,
        internal_params.whir.mu_pow_bits > 0
    );
    // folding_pow_bits affects the number of interactions in SumcheckAir. The app preset may
    // omit folding PoW even when aggregation presets use it: this changes the leaf verifier,
    // whose VK is app-specific, but does not change the internal recursive VK. The leaf and
    // internal presets must still agree so the internal-for-leaf and recursive verifier shapes
    // remain compatible.
    assert_eq!(
        leaf_params.whir.folding_pow_bits > 0,
        internal_params.whir.folding_pow_bits > 0
    );
    // query_phase_pow_bits affects the number of interactions in WhirRoundAir.
    assert_eq!(
        app_params.whir.query_phase_pow_bits > 0,
        leaf_params.whir.query_phase_pow_bits > 0
    );
    assert_eq!(
        leaf_params.whir.query_phase_pow_bits > 0,
        internal_params.whir.query_phase_pow_bits > 0
    );
}

#[cfg(test)]
mod tests {
    use std::{panic::catch_unwind, sync::Arc};

    use openvm_stark_backend::{
        test_utils::{FibFixture, TestFixture},
        StarkEngine,
    };
    use openvm_stark_sdk::config::{
        app_params_with_128_bits_field_security,
        baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge},
        internal_params_with_128_bits_field_security, leaf_params_with_128_bits_field_security,
    };

    use super::check_param_compatibility;
    use crate::system::{AggregationSubCircuit, VerifierSubCircuit};

    #[test]
    fn test_folding_pow_openvm_boundary_contract() {
        let app_params = app_params_with_128_bits_field_security(21);
        let leaf_params = leaf_params_with_128_bits_field_security();
        let internal_params = internal_params_with_128_bits_field_security();

        // These are deliberate production-profile goldens: the app omits folding PoW while both
        // aggregation profiles include it. The app/leaf mismatch must remain admissible.
        assert_eq!(app_params.whir.folding_pow_bits, 0);
        assert_eq!(leaf_params.whir.folding_pow_bits, 1);
        assert_eq!(internal_params.whir.folding_pow_bits, 14);
        check_param_compatibility(&app_params, &leaf_params, &internal_params);

        // Leaf and internal circuits must still agree on whether folding PoW is present because
        // that boolean changes the internal recursive verifier shape.
        let mut incompatible_internal_params = internal_params.clone();
        incompatible_internal_params.whir.folding_pow_bits = 0;
        assert!(
            catch_unwind(|| {
                check_param_compatibility(&app_params, &leaf_params, &incompatible_internal_params);
            })
            .is_err(),
            "leaf/internal folding-PoW mismatch must be rejected"
        );

        // Construct the actual leaf-verifying-app boundary. WhirModule is built from the child
        // (app) VK, so its Sumcheck AIR must use the app's zero folding-PoW value even though the
        // verifier circuit itself is keyed with the leaf profile.
        let app_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(app_params.clone());
        let (_, app_vk) = FibFixture::new(0, 1, 1).keygen(&app_engine);
        let verifier = VerifierSubCircuit::<1>::new(Arc::new(app_vk));
        assert_eq!(
            verifier.whir.folding_pow_bits(),
            app_params.whir.folding_pow_bits
        );

        let leaf_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(leaf_params.clone());
        let (_, leaf_vk) = leaf_engine.keygen(&verifier.airs());
        assert_eq!(
            leaf_vk.inner.params.whir.folding_pow_bits,
            leaf_params.whir.folding_pow_bits
        );
    }
}
