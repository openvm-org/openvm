use afs_stark_backend::interaction::InteractionBuilder;
use p3_keccak_air::NUM_ROUNDS;

use super::{columns::KeccakPermuteCols, KeccakPermuteAir, NUM_U64_HASH_ELEMS};

impl KeccakPermuteAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakPermuteCols<AB::Var>,
    ) {
        let output = (0..5).flat_map(move |x| {
            (0..5).flat_map(move |y| {
                (0..NUM_U64_HASH_ELEMS).map(move |limb| {
                    // TODO: after switching to latest p3 commit, this should be y, x
                    local.keccak.a_prime_prime_prime(x, y, limb)
                })
            })
        });
        let input = local.keccak.preimage.into_iter().flatten().flatten();
        let is_input = local.is_direct * local.keccak.step_flags[0];
        builder.push_receive(self.input_bus, input, is_input);

        let is_output = local.is_direct * local.keccak.step_flags[NUM_ROUNDS - 1];
        builder.push_send(self.output_bus, output, is_output);
    }
}
