use halo2_base::gates::circuit::BaseCircuitParams;
use serde::{Deserialize, Serialize};

pub const STATIC_VERIFIER_NUM_ADVICE_COLS: usize = 1;
pub const STATIC_VERIFIER_LOOKUP_ADVICE_COLS: usize = 1;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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
            lookup_bits: 11,
            minimum_rows: 20,
            instance_columns: 1,
        }
    }
}

impl StaticVerifierShape {
    pub fn expected_phase0_params(&self) -> BaseCircuitParams {
        BaseCircuitParams {
            k: self.k,
            num_advice_per_phase: vec![STATIC_VERIFIER_NUM_ADVICE_COLS],
            num_fixed: 1,
            num_lookup_advice_per_phase: vec![STATIC_VERIFIER_LOOKUP_ADVICE_COLS],
            lookup_bits: Some(self.lookup_bits),
            num_instance_columns: self.instance_columns,
        }
    }
}
