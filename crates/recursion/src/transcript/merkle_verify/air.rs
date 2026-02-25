use core::{array, borrow::Borrow};

pub use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{CHUNK, DIGEST_SIZE};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{
    CommitmentsBus, CommitmentsBusMessage, MerkleVerifyBus, MerkleVerifyBusMessage,
    Poseidon2CompressBus, Poseidon2CompressMessage,
};

/// There are two parts in the merkle proof: hashing leaves and the (standard) merkle proof.
///
/// Example: (k = 2), going from left to right
/// leaf0 \
/// leaf1 -> a
/// leaf2 \    \
/// leaf3 -> b -> c \                 will start hashing with merkle proof siblings
///            merkle_sibing -> ...
///
/// (First part) Hashing leaves: there are `2^k` leaves, each is [T; DIGEST_SIZE]. Each row in
/// MerkleVerify AIR represents a Poseidon2 compression (of two leaves, or two intermediate values).
/// So there are `2^k - 1` rows for this part. At height 0, the leaves are at the bottom (leaf0 ~
/// leaf3) in the diagram above. At height 1 are the intermediate hashes (a, b) in the diagram
/// above. The first row in the AIR will be Poseidon2 compression of leaf0 and leaf1, and the second
/// row will be Poseidon2 compression of leaf2 and leaf3. And the third row will be Poseidon2
/// compression of a and b.
///
/// (Second part) Standard merkle proof, the next row will be Poseidon2 compression of `c` and the
/// sibling of `c`.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MerkleVerifyCols<T> {
    pub proof_idx: T,
    pub is_proof_start: T,
    pub merkle_proof_idx: T,
    pub is_valid: T,
    /// Indicator: whether this is the first row of a merkle proof
    pub is_first_merkle: T,
    /// Indicator: whether this is the last row of a merkle proof
    pub is_last_merkle: T,

    pub is_combining_leaves: T,
    pub leaf_sub_idx: T,

    /// The merkle idx, of the current level
    pub idx: T,
    pub idx_parity: T, // 0 for even, 1 for odd, of the merkle idx
    /// Total depth of the merkle proof including the leaves part = merkle_proof.len() + 1 + k
    pub total_depth: T,
    /// 0 -> total_depth - 1, where leaves are at height 0, combined leaf hash is at height k
    pub height: T,

    pub left: [T; DIGEST_SIZE],
    pub right: [T; DIGEST_SIZE],

    /// Indicator: whether to receive the left/right value
    pub recv_left: T,
    pub recv_right: T,

    pub commit_major: T,
    pub commit_minor: T,

    pub compression_output: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
pub(super) struct MerkleVerifyLog {
    pub merkle_idx: usize,
    pub depth: usize,
    pub query_idx: usize,
    /// For round 0, this is 0. For rounds > 0, this is the round index.
    pub commit_major: usize,
    /// For round 0, this is the commit index. For rounds > 0, this is 0.
    pub commit_minor: usize,
}

pub struct MerkleVerifyAir {
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub merkle_verify_bus: MerkleVerifyBus,
    pub commitments_bus: CommitmentsBus,
}

impl<F: Field> BaseAir<F> for MerkleVerifyAir {
    fn width(&self) -> usize {
        MerkleVerifyCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MerkleVerifyAir {}
impl<F: Field> PartitionedBaseAir<F> for MerkleVerifyAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for MerkleVerifyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &MerkleVerifyCols<AB::Var> = (*local).borrow();
        let next: &MerkleVerifyCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////
        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_proof_start);
        builder.assert_bool(local.is_first_merkle);
        builder.assert_bool(local.is_last_merkle);
        builder.assert_bool(local.is_combining_leaves);
        builder.assert_bool(local.recv_left);
        builder.assert_bool(local.recv_right);

        // At the last row, the cur hash should be consistent with the commit
        for i in 0..DIGEST_SIZE {
            builder
                .when(local.is_last_merkle)
                .assert_eq(local.left[i], local.right[i]);
        }

        // Boundary constraints
        builder
            .when(local.is_first_merkle)
            .assert_zero(local.height);
        builder
            .when(local.is_last_merkle)
            .assert_eq(local.total_depth, local.height + AB::Expr::ONE);

        // transition of idx and depth, during merkle proof part
        let is_merkle_transition = (AB::Expr::ONE - local.is_last_merkle)
            * local.is_valid
            * (AB::Expr::ONE - local.is_combining_leaves);
        builder
            .when(is_merkle_transition.clone())
            .assert_eq(local.height + AB::Expr::ONE, next.height);
        builder.assert_bool(local.idx_parity);
        builder.when(is_merkle_transition.clone()).assert_eq(
            local.idx,
            next.idx * AB::Expr::from_usize(2) + local.idx_parity,
        );

        // Always receive both left and right for combining leaves part, otherwise receive only one
        // of them
        builder
            .when(local.is_combining_leaves)
            .assert_one(local.recv_left);
        builder
            .when(local.is_combining_leaves)
            .assert_one(local.recv_right);
        builder
            .when((AB::Expr::ONE - local.is_combining_leaves) * local.is_valid)
            .assert_one(local.recv_right + local.recv_left);

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // This is 2x / 2x + 1 if it's a combining leaves part, otherwise it's 0.
        let left_leaf_sub_idx =
            local.is_combining_leaves * local.leaf_sub_idx * AB::Expr::from_usize(2);
        let right_leaf_sub_idx = local.is_combining_leaves
            * (local.leaf_sub_idx * AB::Expr::from_usize(2) + AB::Expr::ONE);
        self.merkle_verify_bus.receive(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.left.map(Into::into),
                merkle_idx: local.idx.into(),
                total_depth: local.total_depth.into(),
                height: local.height.into(),
                leaf_sub_idx: left_leaf_sub_idx,
                commit_major: local.commit_major.into(),
                commit_minor: local.commit_minor.into(),
            },
            local.is_valid * local.recv_left,
        );
        self.merkle_verify_bus.receive(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.right.map(Into::into),
                merkle_idx: local.idx.into(),
                total_depth: local.total_depth.into(),
                height: local.height.into(),
                leaf_sub_idx: right_leaf_sub_idx,
                commit_major: local.commit_major.into(),
                commit_minor: local.commit_minor.into(),
            },
            local.is_valid * local.recv_right,
        );
        // At "combining leaves" part, the idx is the same for all the rows.
        // Otherwise the idx should be half of the previous idx, which is just next.idx.
        let send_merkle_idx = local.idx * local.is_combining_leaves
            + (AB::Expr::ONE - local.is_combining_leaves) * next.idx;
        self.merkle_verify_bus.send(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.compression_output.map(Into::into),
                merkle_idx: send_merkle_idx,
                total_depth: local.total_depth.into(),
                height: local.height + AB::Expr::ONE,
                leaf_sub_idx: local.leaf_sub_idx.into(),
                commit_major: local.commit_major.into(),
                commit_minor: local.commit_minor.into(),
            },
            (AB::Expr::ONE - local.is_last_merkle) * local.is_valid,
        );

        self.commitments_bus.lookup_key(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: local.commit_major,
                minor_idx: local.commit_minor,
                commitment: local.left, // left and right are both the commitment
            },
            local.is_valid * local.is_last_merkle,
        );

        let poseidon2_input = array::from_fn(|i| {
            if i < CHUNK {
                local.left[i].into()
            } else {
                local.right[i - CHUNK].into()
            }
        });
        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon2_input,
                output: local.compression_output.map(Into::into),
            },
            local.is_valid,
        );
    }
}

pub(super) fn compute_cum_sum<T>(values: &[T], f: impl Fn(&T) -> usize) -> Vec<usize> {
    values
        .iter()
        .scan(0usize, |acc, x| {
            *acc += f(x);
            Some(*acc)
        })
        .collect()
}
