use std::array::from_fn;

use itertools::Itertools;
use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{utils::not, SubAir};
use openvm_recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::AirBuilder;
use p3_field::PrimeCharacteristicRing;

#[derive(Copy, Clone, Debug)]
pub struct MerklePathRowView<'a, T> {
    // 0 for invalid, 1 for valid, 2 for valid first row
    pub is_valid: &'a T,
    pub is_right_child: &'a T,
    pub node_commit: &'a [T; DIGEST_SIZE],
    pub sibling: &'a [T; DIGEST_SIZE],
    // 2^row_idx, used to (a) constrain merkle proof height and (b) accumulate
    // merkle_path_branch_bits
    pub row_idx_exp_2: &'a T,
    pub merkle_path_branch_bits: &'a T,
}

#[derive(Copy, Clone, Debug)]
pub struct MerklePathSubAirContext<'a, T> {
    pub local: MerklePathRowView<'a, T>,
    pub next: MerklePathRowView<'a, T>,
}

/// SubAir to constrain a single Merkle path represented over consecutive rows.
#[derive(Clone, Debug, derive_new::new)]
pub struct MerklePathSubAir {
    pub poseidon2_bus: Poseidon2CompressBus,
    pub expected_proof_len: usize,
    pub expected_branch_bits: u32,
}

impl<AB: AirBuilder + InteractionBuilder> SubAir<AB> for MerklePathSubAir {
    type AirContext<'a>
        = (
        MerklePathSubAirContext<'a, AB::Var>,
        AB::Expr,
        AB::Expr,
        AB::Expr,
    )
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let local = ctx.0.local;
        let next = ctx.0.next;

        builder.assert_tern(*local.is_valid);
        builder.assert_bool(*local.is_right_child);

        /*
         * Constrain that is_valid is 2 on the first row, that the rest of the Merkle
         * tree immediately follows, and that all invalid rows are at the end.
         */
        builder
            .when_first_row()
            .assert_eq(*local.is_valid, AB::F::TWO);
        builder
            .when_transition()
            .assert_bool(*local.is_valid - *next.is_valid);
        builder.when_transition().assert_bool(*next.is_valid);

        /*
         * Constrain that the Merkle path starts at depth 0 (row_idx_exp_2 = 1), then
         * doubles every valid row. The accumulated branch bits start from is_right_child
         * on the first row and must end at expected_branch_bits.
         */
        builder.when_first_row().assert_one(*local.row_idx_exp_2);
        builder
            .when_first_row()
            .assert_eq(*local.merkle_path_branch_bits, *local.is_right_child);

        let mut when_transition = builder.when_transition();
        let mut when_both_valid = when_transition.when(*next.is_valid);
        when_both_valid.assert_eq(*local.row_idx_exp_2 * AB::F::TWO, *next.row_idx_exp_2);
        when_both_valid.assert_eq(
            *local.merkle_path_branch_bits + *next.is_right_child * *next.row_idx_exp_2,
            *next.merkle_path_branch_bits,
        );

        let expected_branch_bits_offset = ctx.3;
        let mut when_last = builder.when(*local.is_valid * (*next.is_valid - AB::F::ONE));
        when_last.assert_eq(
            *local.merkle_path_branch_bits,
            expected_branch_bits_offset + AB::F::from_u32(self.expected_branch_bits),
        );
        when_last.assert_eq(
            *local.row_idx_exp_2,
            AB::F::from_usize(1 << self.expected_proof_len),
        );
        when_last.assert_zero(*local.is_right_child);

        /*
         * We give the option to skip hashing until some row_idx, effectively allowing
         * the Merkle proof to begin there. The caller MUST constrain the validity of
         * node_commit and row_idx on that row themselves. Additionally, there are some
         * restrictions:
         * - All skips must happen at the beginning
         * - is_valid  must be set as normal
         * - is_right_child and merkle_path_branch_bits must be 0
         */
        let local_skip_hash = ctx.1;
        let next_skip_hash = ctx.2;

        builder.assert_bool(local_skip_hash.clone());
        builder
            .when_transition()
            .assert_bool(local_skip_hash.clone() - next_skip_hash.clone());
        builder
            .when_transition()
            .when(local_skip_hash.clone() - next_skip_hash)
            .assert_one(*next.is_valid);

        let mut when_skip_hash = builder.when(local_skip_hash.clone());
        when_skip_hash.assert_zero(*local.is_right_child);
        when_skip_hash.assert_zero(*local.merkle_path_branch_bits);

        /*
         * Constrain that next.node_commit is the Poseidon2 compression of local_commit
         * and sibling. Note that node_commit on the first row is unconstrained.
         */
        let left: [AB::Expr; DIGEST_SIZE] = from_fn(|i| {
            (AB::Expr::ONE - *local.is_right_child) * local.node_commit[i]
                + *local.is_right_child * local.sibling[i]
        });
        let right: [AB::Expr; DIGEST_SIZE] = from_fn(|i| {
            *local.is_right_child * local.node_commit[i]
                + (AB::Expr::ONE - *local.is_right_child) * local.sibling[i]
        });

        let poseidon2_input: [AB::Expr; POSEIDON2_WIDTH] =
            left.into_iter().chain(right).collect_array().unwrap();
        let should_hash = *next.is_valid * (AB::Expr::TWO - *next.is_valid);

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon2_input,
                output: next.node_commit.map(Into::into),
            },
            should_hash * not::<AB::Expr>(local_skip_hash),
        );
    }
}
