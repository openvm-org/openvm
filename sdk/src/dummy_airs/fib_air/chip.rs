use std::sync::Arc;

use afs_stark_backend::{
    prover::types::{AirProofInput, AirProofRawInput},
    rap::AnyRap,
    Chip,
};
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{air::FibonacciAir, trace::generate_trace_rows};

pub struct FibonacciChip {
    /// The 0th number in the fibonacci sequence.
    pub a: u32,
    /// The 1st number in the fibonacci sequence.
    pub b: u32,
    /// Target n-th number in the fibonacci sequence.
    pub n: usize,
}

impl FibonacciChip {
    pub fn new(a: u32, b: u32, n: usize) -> Self {
        assert!(n.is_power_of_two());
        Self { a, b, n }
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for FibonacciChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(FibonacciAir)
    }

    fn generate_air_proof_input(&self) -> AirProofInput<SC> {
        let common_main = generate_trace_rows::<Val<SC>>(self.a, self.b, self.n);
        let a = common_main.get(0, 0);
        let b = common_main.get(0, 1);
        let last_val = common_main.get(self.n - 1, 1);
        AirProofInput {
            air: self.air(),
            cached_mains_pdata: vec![],
            raw: AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(generate_trace_rows::<Val<SC>>(self.a, self.b, self.n)),
                public_values: vec![a, b, last_val],
            },
        }
    }
}
