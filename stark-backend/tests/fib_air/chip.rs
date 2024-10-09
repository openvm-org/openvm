use afs_stark_backend::{
    prover::v2::types::{AirProofInput, Chip},
    rap::AnyRap,
};
use p3_field::{AbstractField, PrimeField32};
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::fib_air::{air::FibonacciAir, trace::generate_trace_rows};

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
    fn air(&self) -> &dyn AnyRap<SC> {
        &FibonacciAir
    }

    fn generate_air_proof_input(&self) -> AirProofInput<'_, SC> {
        AirProofInput {
            air: self.air(),
            cached_mains: vec![],
            common_main: Some(generate_trace_rows::<Val<SC>>(self.a, self.b, self.n)),
            public_values: [self.a, self.b, get_fib_number(self.n)]
                .into_iter()
                .map(Val::<SC>::from_canonical_u32)
                .collect(),
        }
    }
}

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}
