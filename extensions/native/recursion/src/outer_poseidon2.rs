use openvm_native_compiler::ir::{Builder, Config, DslIr, Felt, Var};
use openvm_stark_backend::p3_field::{
    max_shifted_absorb_injective_limbs, Field, PrimeCharacteristicRing,
};

use crate::{
    utils::{absorb_radix_bits_for_config, reduce_packed_shifted},
    vars::OuterDigestVariable,
    OUTER_DIGEST_SIZE,
};

pub const SPONGE_SIZE: usize = 3;
pub const RATE: usize = 2;
const POSEIDON_CELL_TRACKER_NAME: &str = "PoseidonCell";

pub trait Poseidon2CircuitBuilder<C: Config> {
    fn p2_permute_mut(&mut self, state: [Var<C::N>; SPONGE_SIZE]);
    #[allow(dead_code)]
    fn p2_hash(&mut self, input: &[Felt<C::F>]) -> OuterDigestVariable<C>;
    #[allow(dead_code)]
    fn p2_compress(&mut self, input: [OuterDigestVariable<C>; RATE]) -> OuterDigestVariable<C>;
}

impl<C: Config> Poseidon2CircuitBuilder<C> for Builder<C> {
    fn p2_permute_mut(&mut self, state: [Var<C::N>; SPONGE_SIZE]) {
        self.cycle_tracker_start(POSEIDON_CELL_TRACKER_NAME);
        p2_permute_mut_impl(self, state);
        self.cycle_tracker_end(POSEIDON_CELL_TRACKER_NAME);
    }

    fn p2_hash(&mut self, input: &[Felt<C::F>]) -> OuterDigestVariable<C> {
        self.cycle_tracker_start(POSEIDON_CELL_TRACKER_NAME);
        assert_eq!(C::N::bits(), openvm_stark_sdk::p3_bn254::Bn254::bits());
        assert_eq!(
            C::F::bits(),
            openvm_stark_sdk::p3_baby_bear::BabyBear::bits()
        );
        let mut state: [Var<C::N>; SPONGE_SIZE] = [
            self.eval(C::N::ZERO),
            self.eval(C::N::ZERO),
            self.eval(C::N::ZERO),
        ];
        // Mirrors `MultiField32PaddingFreeSponge::hash_iter`: chunks of
        // `RATE * num_f_elms` F values per permutation, packing each `num_f_elms` chunk into
        // one PF rate slot via `reduce_packed_shifted`.
        let num_f_elms = max_shifted_absorb_injective_limbs::<C::F, C::N>();
        let radix_bits = absorb_radix_bits_for_config::<C>();
        for block_chunk in input.chunks(RATE * num_f_elms) {
            for (chunk_id, chunk) in block_chunk.chunks(num_f_elms).enumerate() {
                let packed = reduce_packed_shifted(self, chunk, radix_bits);
                // Wrap in builder.eval to give state[chunk_id] a fresh Var: the halo2 lowering
                // of CircuitPoseidon2Permute rebinds each state Var's value to the post-perm
                // result, so aliasing the `packed` Var would corrupt it on subsequent uses.
                state[chunk_id] = self.eval(packed);
            }
            p2_permute_mut_impl(self, state);
        }
        self.cycle_tracker_end(POSEIDON_CELL_TRACKER_NAME);

        [state[0]]
    }

    fn p2_compress(&mut self, input: [OuterDigestVariable<C>; 2]) -> OuterDigestVariable<C> {
        self.cycle_tracker_start(POSEIDON_CELL_TRACKER_NAME);
        let state: [Var<C::N>; SPONGE_SIZE] = [
            self.eval(input[0][0]),
            self.eval(input[1][0]),
            self.eval(C::N::ZERO),
        ];
        p2_permute_mut_impl(self, state);
        self.cycle_tracker_end(POSEIDON_CELL_TRACKER_NAME);
        [state[0]; OUTER_DIGEST_SIZE]
    }
}

fn p2_permute_mut_impl<C: Config>(builder: &mut Builder<C>, state: [Var<C::N>; SPONGE_SIZE]) {
    builder.push(DslIr::CircuitPoseidon2Permute(state))
}
