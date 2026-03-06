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
    /// Index of the proof this row is for
    pub proof_idx: T,
    /// Indicator: whether this row is valid
    pub is_valid: T,
    /// Indicator: whether this is the last row of a merkle proof
    pub is_last_merkle: T,

    /// Indicator: whether this row hashes two leaves or is part of the Merkle path proof
    pub is_combining_leaves: T,
    /// Indicator: whether this row is the root of the hashed leaves
    pub is_last_leaf: T,
    /// Row node index of this node in the leaf hash tree, 0 if is_combining_leaves is false
    pub leaf_sub_idx: T,

    /// The merkle idx of the leaf hash root
    pub idx: T,
    /// Total depth of the merkle proof including the leaves part = merkle_proof.len() + 1 + k
    pub total_depth: T,
    /// 0 -> total_depth - 1, where leaves are at height 0, combined leaf hash is at height k
    pub height: T,

    pub left: [T; DIGEST_SIZE],
    pub right: [T; DIGEST_SIZE],

    /// Flag that indicates what to receive; 0 for left, 1 for right, 2 for both
    pub recv_flag: T,

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
    pub k: usize,
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
        let local = main.row_slice(0).expect("window should have two elements");
        let local: &MerkleVerifyCols<AB::Var> = (*local).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////
        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_last_merkle);
        builder.assert_bool(local.is_combining_leaves);
        builder.assert_bool(local.is_last_leaf);
        builder.assert_tern(local.recv_flag);

        // At the last row, the cur hash should be consistent with the commit
        for i in 0..DIGEST_SIZE {
            builder
                .when(local.is_last_merkle)
                .assert_eq(local.left[i], local.right[i]);
        }

        // Last row for a Merkle tree should have height + 1 == total_depth
        builder
            .when(local.is_last_merkle)
            .assert_eq(local.height + AB::Expr::ONE, local.total_depth);

        // When is_combining_leaves is false, leaf_sub_idx must be 0
        builder
            .when(AB::Expr::ONE - local.is_combining_leaves)
            .assert_zero(local.leaf_sub_idx);

        // On the last leaf, leaf_sub_idx must be 0 and height + 1 must be k
        builder
            .when(local.is_last_leaf)
            .assert_one(local.is_combining_leaves);
        builder
            .when(local.is_last_leaf)
            .assert_zero(local.leaf_sub_idx);
        builder
            .when(local.is_last_leaf)
            .assert_eq(local.height + AB::Expr::ONE, AB::Expr::from_usize(self.k));

        // Receive both left and right for combining leaves part, otherwise receive only one
        builder
            .when(local.is_combining_leaves)
            .assert_one(local.is_valid);
        builder
            .when(local.is_combining_leaves)
            .assert_eq(local.recv_flag, AB::Expr::TWO);
        builder
            .when_ne(local.is_combining_leaves, AB::Expr::ONE)
            .assert_bool(local.recv_flag);

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // This is 2x / 2x + 1 if it's a combining leaves part, otherwise it's 0.
        let left_leaf_sub_idx = local.is_combining_leaves * local.leaf_sub_idx * AB::Expr::TWO;
        let right_leaf_sub_idx =
            local.is_combining_leaves * (local.leaf_sub_idx * AB::Expr::TWO + AB::Expr::ONE);

        let recv_left = (local.recv_flag - AB::Expr::ONE).square();
        let recv_right =
            local.recv_flag * (AB::Expr::from_u8(3) - local.recv_flag) * AB::F::TWO.inverse();

        self.merkle_verify_bus.receive(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.left.map(Into::into),
                merkle_idx: local.idx.into(),
                total_depth: local.total_depth.into(),
                height: local.height.into(),
                is_leaf: local.is_combining_leaves.into(),
                leaf_sub_idx: left_leaf_sub_idx,
                commit_major: local.commit_major.into(),
                commit_minor: local.commit_minor.into(),
            },
            local.is_valid * recv_left,
        );
        self.merkle_verify_bus.receive(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.right.map(Into::into),
                merkle_idx: local.idx.into(),
                total_depth: local.total_depth.into(),
                height: local.height.into(),
                is_leaf: local.is_combining_leaves.into(),
                leaf_sub_idx: right_leaf_sub_idx,
                commit_major: local.commit_major.into(),
                commit_minor: local.commit_minor.into(),
            },
            local.is_valid * recv_right,
        );

        self.merkle_verify_bus.send(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.compression_output.map(Into::into),
                merkle_idx: local.idx.into(),
                total_depth: local.total_depth.into(),
                height: local.height + AB::Expr::ONE,
                is_leaf: local.is_combining_leaves - local.is_last_leaf,
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
