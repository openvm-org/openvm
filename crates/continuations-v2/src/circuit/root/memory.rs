use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::PUBLIC_VALUES_AS},
};
use openvm_circuit_primitives::utils::not;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_util::log2_strict_usize,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::circuit::root::{
    bus::{
        MemoryMerkleCommitBus, MemoryMerkleCommitMessage, UserPvsCommitBus, UserPvsCommitMessage,
    },
    digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UserPvsInMemoryCols<F> {
    // 0 for invalid, 1 for valid, 2 for valid first row
    pub is_valid: F,
    pub is_right_child: F,
    pub node_commit: [F; DIGEST_SIZE],
    pub sibling: [F; DIGEST_SIZE],

    // 2^row_idx, used to (a) constrain merkle proof height and (b) accumulate
    // merkle_path_branch_bits
    pub row_idx_exp_2: F,
    pub merkle_path_branch_bits: F,
}

pub struct UserPvsInMemoryAir {
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub memory_merkle_commit_bus: MemoryMerkleCommitBus,

    // Used to constrain that the user public values reside in the correct address space
    expected_proof_len: usize,
    merkle_path_branch_bits: u32,
}

impl UserPvsInMemoryAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        memory_merkle_commit_bus: MemoryMerkleCommitBus,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self {
        assert!(memory_dimensions.addr_space_height > 1);
        let pv_start_idx = memory_dimensions.label_to_index((PUBLIC_VALUES_AS, 0));
        let pv_height = log2_strict_usize(num_user_pvs / DIGEST_SIZE);
        let merkle_path_branch_bits = u32::try_from(pv_start_idx >> pv_height)
            .expect("merkle_path_branch_bits must fit in u32");
        let expected_proof_len = memory_dimensions.overall_height() - pv_height;
        Self {
            poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            expected_proof_len,
            merkle_path_branch_bits,
        }
    }
}

impl<F> BaseAir<F> for UserPvsInMemoryAir {
    fn width(&self) -> usize {
        UserPvsInMemoryCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsInMemoryAir {}
impl<F> PartitionedBaseAir<F> for UserPvsInMemoryAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UserPvsInMemoryAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &UserPvsInMemoryCols<AB::Var> = (*local).borrow();
        let next: &UserPvsInMemoryCols<AB::Var> = (*next).borrow();

        builder.assert_tern(local.is_valid);
        builder.assert_bool(local.is_right_child);

        /*
         * Constrain that is_valid is 2 on the first row, that the rest of the Merkle
         * tree immediately follows, and that all invalid rows are at the end.
         */
        builder
            .when_first_row()
            .assert_eq(local.is_valid, AB::F::TWO);
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);
        builder.when_transition().assert_bool(next.is_valid);

        /*
         * Receive the user public values commit on the first row. The first DIGEST_SIZE
         * elements of perm state are this merkle tree node's commit.
         */
        self.user_pvs_commit_bus.receive(
            builder,
            UserPvsCommitMessage {
                user_pvs_commit: local.node_commit,
            },
            local.is_valid * (local.is_valid - AB::F::ONE) * AB::F::TWO.inverse(),
        );

        /*
         * Constrain that the Merkle path is the shape it should be. There should be
         * addr_space_height + 1 valid rows, and the sum of is_right_child * 2^row_idx
         * over valid rows should match merkle_path_branch_bits.
         */
        builder.when_first_row().assert_one(local.row_idx_exp_2);
        builder
            .when_first_row()
            .assert_eq(local.merkle_path_branch_bits, local.is_right_child);

        let mut when_transition = builder.when_transition();
        let mut when_both_valid = when_transition.when(next.is_valid);
        when_both_valid.assert_eq(local.row_idx_exp_2 * AB::F::TWO, next.row_idx_exp_2);
        when_both_valid.assert_eq(
            local.merkle_path_branch_bits + next.is_right_child * next.row_idx_exp_2,
            next.merkle_path_branch_bits,
        );

        let mut when_last = builder.when(local.is_valid * (next.is_valid - AB::F::ONE));
        when_last.assert_eq(
            local.merkle_path_branch_bits,
            AB::F::from_u32(self.merkle_path_branch_bits),
        );
        when_last.assert_eq(
            local.row_idx_exp_2,
            AB::F::from_usize(1 << self.expected_proof_len),
        );
        when_last.assert_zero(local.is_right_child);

        /*
         * Constrain that next.perm_state is the Poseidon2 permutation of local_commit
         * and sibling. Note that perm_state of the first row is unconstrained.
         */
        let left: [AB::Expr; DIGEST_SIZE] = from_fn(|i| {
            not(local.is_right_child) * local.node_commit[i]
                + local.is_right_child * local.sibling[i]
        });
        let right: [AB::Expr; DIGEST_SIZE] = from_fn(|i| {
            local.is_right_child * local.node_commit[i]
                + not(local.is_right_child) * local.sibling[i]
        });

        let poseidon2_input: [AB::Expr; POSEIDON2_WIDTH] =
            left.into_iter().chain(right).collect_array().unwrap();
        let should_hash = next.is_valid * (AB::Expr::TWO - next.is_valid);

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon2_input,
                output: next.node_commit.map(Into::into),
            },
            should_hash,
        );

        /*
         * Receive the final memory merkle root on the last valid row.
         */
        self.memory_merkle_commit_bus.receive(
            builder,
            MemoryMerkleCommitMessage {
                merkle_root: local.node_commit,
            },
            local.is_valid * (AB::Expr::ONE - next.is_valid).square(),
        );
    }
}

pub fn generate_proving_input(
    user_pv_commit: [F; DIGEST_SIZE],
    merkle_proof: &[[F; DIGEST_SIZE]],
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    let merkle_proof_len = merkle_proof.len();
    let num_layers = merkle_proof_len + 1;
    let height = num_layers.next_power_of_two();
    let width = UserPvsInMemoryCols::<u8>::width();

    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);
    let mut current = user_pv_commit;

    /*
     * We can determine the public values' location in memory (and thus location in
     * the memory merkle tree) from PUBLIC_VALUES_AS, the memory dimensions, and the
     * number of user public values.
     */
    let pv_start_idx = memory_dimensions.label_to_index((PUBLIC_VALUES_AS, 0));
    let pv_height = log2_strict_usize(num_user_pvs / DIGEST_SIZE);
    let merkle_path_branch_bits = pv_start_idx >> pv_height;
    let mut current_branch_bits = 0;

    let mut poseidon2_compress_inputs = Vec::with_capacity(merkle_proof_len);

    for (i, &sibling) in merkle_proof.iter().enumerate() {
        let chunk = chunks.next().unwrap();
        let cols: &mut UserPvsInMemoryCols<F> = chunk.borrow_mut();
        let is_right_child = merkle_path_branch_bits & (1 << i) != 0;
        current_branch_bits += (is_right_child as usize) << i;

        cols.is_valid = if i == 0 { F::TWO } else { F::ONE };
        cols.is_right_child = F::from_bool(is_right_child);
        cols.node_commit = current;
        cols.sibling = sibling;
        cols.row_idx_exp_2 = F::from_usize(1 << i);
        cols.merkle_path_branch_bits = F::from_usize(current_branch_bits);

        let left = if is_right_child { sibling } else { current };
        let right = if is_right_child { current } else { sibling };
        current = poseidon2_compress_with_capacity(left, right).0;
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));
    }

    let last_chunk = chunks.next().unwrap();
    let last_row: &mut UserPvsInMemoryCols<F> = last_chunk.borrow_mut();
    last_row.is_valid = F::ONE;
    last_row.node_commit = current;
    last_row.row_idx_exp_2 = F::from_usize(1 << merkle_proof_len);
    last_row.merkle_path_branch_bits = F::from_usize(current_branch_bits);

    (
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&RowMajorMatrix::new(
            trace, width,
        ))),
        poseidon2_compress_inputs,
    )
}
