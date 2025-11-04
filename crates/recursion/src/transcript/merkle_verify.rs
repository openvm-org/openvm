use core::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::Matrix;
use stark_backend_v2::{DIGEST_SIZE, F, poseidon2::sponge::poseidon2_compress, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{CommitmentsBus, CommitmentsBusMessage, MerkleVerifyBus, Poseidon2Bus},
    system::Preflight,
};

pub const CHUNK: usize = 8;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MerkleVerifyCols<T> {
    pub proof_idx: T,
    pub merkle_proof_idx: T,
    pub is_valid: T,
    /// Indicator: whether this is the first row of a merkle proof
    pub is_first_merkle: T,
    /// Indicator: whether this is the last row of a merkle proof
    pub is_last_merkle: T, // TODO: do we need this?

    /// The merkle idx, of the current level
    pub idx: T,
    pub idx_parity: T, // 0 for even, 1 for odd, of the merkle idx
    /// The current depth of the merkle proof
    pub depth: T,

    pub cur_hash: [T; DIGEST_SIZE],

    pub commit_major: T,
    pub commit_minor: T,

    // it's the commit if is_last_merkle, otherwise it's the sibling merkle proof
    pub proof_or_commit: [T; DIGEST_SIZE],
}

pub struct MerkleVerifyAir {
    pub poseidon2_bus: Poseidon2Bus,
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
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MerkleVerifyCols<AB::Var> = (*local).borrow();
        let _next: &MerkleVerifyCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        // At the last row, the cur hash should be consistent with the commit
        for i in 0..DIGEST_SIZE {
            builder
                .when(local.is_last_merkle)
                .assert_eq(local.proof_or_commit[i], local.cur_hash[i]);
        }

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // TODO: enable this when WHIR's OpenedValuesAir send
        // self.merkle_verify_bus.receive(
        //     builder,
        //     local.proof_idx,
        //     local.merkle_verify_bus_msg.clone(),
        //     local.is_valid,
        // );
        self.commitments_bus.receive(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: local.commit_major,
                minor_idx: local.commit_minor,
                commitment: local.proof_or_commit,
            },
            local.is_last_merkle,
        );
    }
}

fn compute_cum_sum<T>(values: &[T], f: impl Fn(&T) -> usize) -> Vec<usize> {
    values
        .iter()
        .scan(0usize, |acc, x| {
            *acc += f(x);
            Some(*acc)
        })
        .collect()
}

pub fn generate_trace(proofs: &[Proof], preflights: &[Preflight]) -> Vec<F> {
    let width = MerkleVerifyCols::<F>::width();
    // vec of vec, index by proof and then merkle_proof
    let num_rows_per_proof_per_merkle = preflights
        .iter()
        .map(|preflight| {
            preflight
                .merkle_verify_logs
                .iter()
                .map(|log| log.depth + 1)
                .collect_vec()
        })
        .collect_vec();
    // cum sum of proof
    let num_rows_cum_sums: Vec<usize> =
        compute_cum_sum(&num_rows_per_proof_per_merkle, |x| x.iter().sum::<usize>());
    // for each proof, cum sum of merkle proofs
    let num_rows_cum_sums_within_proof: Vec<Vec<usize>> = num_rows_per_proof_per_merkle
        .iter()
        .map(|x| compute_cum_sum(x, |y| *y))
        .collect();
    let num_valid_rows = *num_rows_cum_sums.last().unwrap();
    let height = num_valid_rows.next_power_of_two();
    let mut trace = vec![F::ZERO; height * width];
    let mut cur_hash = [F::ZERO; DIGEST_SIZE];
    let mut cur_idx = 0;
    for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let proof_idx = num_rows_cum_sums.partition_point(|&x| x <= row_idx);
        let preflight = &preflights[proof_idx];
        let proof = &proofs[proof_idx];
        let idx_in_proof = if proof_idx == 0 {
            row_idx
        } else {
            row_idx - num_rows_cum_sums[proof_idx - 1]
        };
        let merkle_proof_idx =
            num_rows_cum_sums_within_proof[proof_idx].partition_point(|&x| x <= idx_in_proof);

        // i: the index (0~depth) within a merkle proof
        let i = if merkle_proof_idx == 0 {
            idx_in_proof
        } else {
            idx_in_proof - num_rows_cum_sums_within_proof[proof_idx][merkle_proof_idx - 1]
        };

        let log = &preflight.merkle_verify_logs[merkle_proof_idx];

        let cols: &mut MerkleVerifyCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_canonical_usize(proof_idx);
        cols.merkle_proof_idx = F::from_canonical_usize(merkle_proof_idx);
        cols.commit_major = F::from_canonical_usize(log.commit_major);
        cols.commit_minor = F::from_canonical_usize(log.commit_minor);
        cols.depth = F::from_canonical_usize(log.depth);
        if i == log.depth {
            cols.is_last_merkle = F::ONE;
        }
        if i == 0 {
            cols.is_first_merkle = F::ONE;
            cur_hash = log.leaf_hash;
            cur_idx = log.merkle_idx;
        }
        cols.cur_hash = cur_hash;
        cols.idx = F::from_canonical_usize(cur_idx);
        cols.idx_parity = F::from_canonical_usize(cur_idx % 2);
        if i < log.depth {
            cols.proof_or_commit =
                proof.whir_proof.codeword_merkle_proofs[log.commit_major - 1][log.query_idx][i];
        } else {
            // last row, it's the commit
            cols.proof_or_commit = proof.whir_proof.codeword_commits[log.commit_major - 1];
        }

        cur_hash = if cur_idx % 2 == 0 {
            poseidon2_compress(cur_hash, cols.proof_or_commit)
        } else {
            poseidon2_compress(cols.proof_or_commit, cur_hash)
        };
        cur_idx /= 2;
    }

    trace
}
