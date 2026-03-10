use std::array::from_fn;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::AirBuilder;
use p3_field::PrimeCharacteristicRing;
use recursion_circuit::bus::{
    Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus, Poseidon2PermuteMessage,
};

use crate::utils::digests_to_poseidon2_input;

/// SubAir to hash N digest-sized elements into a single digest using a chain of Poseidon2
/// operations: N−1 permutations followed by 1 compression. Each step absorbs one element
/// into the capacity portion of the running state.
#[derive(Clone, Debug, derive_new::new)]
pub struct HashSliceSubAir {
    pub compress_bus: Poseidon2CompressBus,
    pub permute_bus: Poseidon2PermuteBus,
}

pub struct HashSliceCtx<'a, T> {
    // N elements
    pub elements: &'a [[T; DIGEST_SIZE]],
    // N - 1 intermediate states
    pub intermediate: &'a [[T; POSEIDON2_WIDTH]],
    // 1 final result
    pub result: &'a [T; DIGEST_SIZE],
    // Boolean value that is true iff enabled
    pub enabled: &'a T,
}

impl<AB: AirBuilder + InteractionBuilder> SubAir<AB> for HashSliceSubAir {
    type AirContext<'a>
        = HashSliceCtx<'a, AB::Expr>
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let n = ctx.elements.len();

        assert_eq!(n, ctx.intermediate.len() + 1);
        assert!(n > 1);

        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: digests_to_poseidon2_input(
                    ctx.elements[0].clone(),
                    [AB::Expr::ZERO; DIGEST_SIZE],
                ),
                output: ctx.intermediate[0].clone(),
            },
            ctx.enabled.clone(),
        );

        for i in 1..(n - 1) {
            self.permute_bus.lookup_key(
                builder,
                Poseidon2PermuteMessage {
                    input: digests_to_poseidon2_input(
                        ctx.elements[i].clone(),
                        from_fn(|j| ctx.intermediate[i - 1][j + DIGEST_SIZE].clone()),
                    ),
                    output: ctx.intermediate[i].clone(),
                },
                ctx.enabled.clone(),
            );
        }

        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    ctx.elements[n - 1].clone(),
                    from_fn(|j| ctx.intermediate[n - 2][j + DIGEST_SIZE].clone()),
                ),
                output: ctx.result.clone(),
            },
            ctx.enabled.clone(),
        );
    }
}
