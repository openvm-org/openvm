use afs_stark_backend::interaction::InteractionBuilder;

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
        builder.push_send(self.bus_output, output, local.is_real_output);

        let input = local.keccak.preimage.into_iter().flatten().flatten();
        builder.push_receive(self.bus_input, input, local.is_real_input);
    }
}
