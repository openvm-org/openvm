use afs_compiler::ir::{Builder, Config, DslIr, Felt, Var};
use itertools::Itertools;
use p3_field::{AbstractField, Field};

use crate::{types::OuterDigestVariable, utils::reduce_32, OUTER_DIGEST_SIZE};

pub const SPONGE_SIZE: usize = 3;
pub const RATE: usize = 2;

pub trait Poseidon2CircuitBuilder<C: Config> {
    fn p2_permute_mut(&mut self, state: [Var<C::N>; SPONGE_SIZE]);
    #[allow(dead_code)]
    fn p2_hash(&mut self, input: &[Felt<C::F>]) -> OuterDigestVariable<C>;
    #[allow(dead_code)]
    fn p2_compress(&mut self, input: [OuterDigestVariable<C>; RATE]) -> OuterDigestVariable<C>;
}

impl<C: Config> Poseidon2CircuitBuilder<C> for Builder<C> {
    fn p2_permute_mut(&mut self, state: [Var<C::N>; SPONGE_SIZE]) {
        self.push(DslIr::CircuitPoseidon2Permute(state))
    }

    fn p2_hash(&mut self, input: &[Felt<C::F>]) -> OuterDigestVariable<C> {
        assert_eq!(C::N::bits(), p3_bn254_fr::Bn254Fr::bits());
        assert_eq!(C::F::bits(), p3_baby_bear::BabyBear::bits());
        let num_f_elms = C::N::bits() / C::F::bits();
        let mut state: [Var<C::N>; SPONGE_SIZE] = [
            self.eval(C::N::zero()),
            self.eval(C::N::zero()),
            self.eval(C::N::zero()),
        ];
        for block_chunk in &input.iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(num_f_elms)).into_iter().enumerate() {
                let chunk = chunk.collect_vec().into_iter().copied().collect::<Vec<_>>();
                state[chunk_id] = reduce_32(self, chunk.as_slice());
            }
            self.p2_permute_mut(state);
        }

        [state[0]]
    }

    fn p2_compress(&mut self, input: [OuterDigestVariable<C>; 2]) -> OuterDigestVariable<C> {
        let state: [Var<C::N>; SPONGE_SIZE] = [
            self.eval(input[0][0]),
            self.eval(input[1][0]),
            self.eval(C::N::zero()),
        ];
        self.p2_permute_mut(state);
        [state[0]; OUTER_DIGEST_SIZE]
    }
}
