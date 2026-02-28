use halo2_base::gates::circuit::BaseCircuitParams;

pub const STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0: usize = 1;
pub const STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0: usize = 1;

#[derive(Clone, Copy, Debug)]
pub struct StaticVerifierShape {
    pub k: usize,
    pub lookup_bits: usize,
    pub minimum_rows: usize,
    pub instance_columns: usize,
}

impl Default for StaticVerifierShape {
    fn default() -> Self {
        Self {
            k: 12,
            lookup_bits: 8,
            minimum_rows: 20,
            instance_columns: 1,
        }
    }
}

impl StaticVerifierShape {
    pub fn expected_phase0_params(&self) -> BaseCircuitParams {
        BaseCircuitParams {
            k: self.k,
            num_advice_per_phase: vec![STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0],
            num_fixed: 1,
            num_lookup_advice_per_phase: vec![STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0],
            lookup_bits: Some(self.lookup_bits),
            num_instance_columns: self.instance_columns,
        }
    }
}
