use std::borrow::Borrow;

use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::PUBLIC_VALUES_AS,
};
use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_util::log2_strict_usize, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use recursion_circuit::bus::Poseidon2CompressBus;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::circuit::{
    root::bus::{
        MemoryMerkleCommitBus, MemoryMerkleCommitMessage, UserPvsCommitBus, UserPvsCommitMessage,
    },
    subair::{MerklePathRowView, MerklePathSubAir, MerklePathSubAirContext},
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
    pub merkle_path_subair: MerklePathSubAir,
    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub memory_merkle_commit_bus: MemoryMerkleCommitBus,
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
            merkle_path_subair: MerklePathSubAir::new(
                poseidon2_compress_bus,
                expected_proof_len,
                merkle_path_branch_bits,
            ),
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
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

        self.merkle_path_subair.eval(
            builder,
            (
                MerklePathSubAirContext {
                    local: MerklePathRowView {
                        is_valid: &local.is_valid,
                        is_right_child: &local.is_right_child,
                        node_commit: &local.node_commit,
                        sibling: &local.sibling,
                        row_idx_exp_2: &local.row_idx_exp_2,
                        merkle_path_branch_bits: &local.merkle_path_branch_bits,
                    },
                    next: MerklePathRowView {
                        is_valid: &next.is_valid,
                        is_right_child: &next.is_right_child,
                        node_commit: &next.node_commit,
                        sibling: &next.sibling,
                        row_idx_exp_2: &next.row_idx_exp_2,
                        merkle_path_branch_bits: &next.merkle_path_branch_bits,
                    },
                },
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ),
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
