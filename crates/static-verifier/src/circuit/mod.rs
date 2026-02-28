use halo2_base::gates::{
    circuit::{BaseCircuitParams, CircuitBuilderStage, builder::BaseCircuitBuilder},
    range::RangeInstructions,
};
pub use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

use crate::config::{
    STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
    StaticVerifierShape,
};

#[derive(Clone, Debug, Default)]
pub struct StaticVerifierCircuit {
    pub shape: StaticVerifierShape,
}

impl StaticVerifierCircuit {
    pub fn builder(&self, stage: CircuitBuilderStage) -> BaseCircuitBuilder<Fr> {
        BaseCircuitBuilder::from_stage(stage)
            .use_k(self.shape.k)
            .use_lookup_bits(self.shape.lookup_bits)
            .use_instance_columns(self.shape.instance_columns)
    }

    pub fn prepare_mock_builder(&self) -> (BaseCircuitBuilder<Fr>, BaseCircuitParams) {
        let mut builder = self.builder(CircuitBuilderStage::Mock);
        {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let witness = ctx.load_witness(Fr::from(7u64));
            range.range_check(ctx, witness, 4);
        }
        let params = builder.calculate_params(Some(self.shape.minimum_rows));
        (builder, params)
    }

    pub fn assert_phase0_shape(params: &BaseCircuitParams) {
        let advice_phase0 = params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default();
        let lookup_phase0 = params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default();
        assert_eq!(
            advice_phase0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
            "phase-0 advice column count mismatch"
        );
        assert_eq!(
            lookup_phase0, STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0,
            "phase-0 lookup-advice column count mismatch"
        );
    }
}

#[cfg(test)]
mod tests;
