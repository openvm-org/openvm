use core::{
    array,
    borrow::{Borrow, BorrowMut},
};

use itertools::Itertools;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::InteractionBuilder, keygen::types::MultiStarkVerifyingKey, proof::Proof,
    BaseAirWithPublicValues, PartitionedBaseAir, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, CHUNK, DIGEST_SIZE, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        CommitmentsBus, CommitmentsBusMessage, MerkleVerifyBus, MerkleVerifyBusMessage,
        Poseidon2CompressBus, Poseidon2CompressMessage,
    },
    system::Preflight,
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
struct MerkleVerifyLog {
    merkle_idx: usize,
    depth: usize,
    query_idx: usize,
    /// For round 0, this is 0. For rounds > 0, this is the round index.
    commit_major: usize,
    /// For round 0, this is the commit index. For rounds > 0, this is 0.
    commit_minor: usize,
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

        self.commitments_bus.receive(
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

fn compute_cum_sum<T>(values: &[T], f: impl Fn(&T) -> usize) -> Vec<usize> {
    values
        .iter()
        .scan(0usize, |acc, x| {
            *acc += f(x);
            Some(*acc)
        })
        .collect()
}

#[tracing::instrument(level = "trace", skip_all)]
pub fn generate_trace(
    mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    proofs: &[Proof<BabyBearPoseidon2Config>],
    preflights: &[Preflight],
    params: &SystemParams,
    required_height: Option<usize>,
) -> Option<(Vec<F>, Vec<[F; POSEIDON2_WIDTH]>)> {
    let k = params.k_whir();
    let width = MerkleVerifyCols::<F>::width();
    let num_leaves: usize = 1 << k;
    let mut poseidon2_compress_inputs = vec![];
    let logs_per_proof = preflights
        .iter()
        .map(|preflight| build_merkle_logs(preflight, params))
        .collect_vec();
    // vec of vec, index by proof and then merkle_proof
    let num_rows_per_proof_per_merkle = preflights
        .iter()
        .enumerate()
        .map(|(proof_idx, _)| {
            logs_per_proof[proof_idx]
                .iter()
                .map(|log| log.depth + num_leaves) // (d + 1) + (num_leaves - 1)
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
    let height = if let Some(height) = required_height {
        if num_valid_rows > height {
            return None;
        }
        height
    } else {
        num_valid_rows.next_power_of_two()
    };
    let mut trace = vec![F::ZERO; height * width];
    let mut cur_hash = [F::ZERO; DIGEST_SIZE];
    let mut cur_idx = 0;

    // layer 0: num_leaves = 2^k hashes
    // layer 1: half the hashes
    // ... layer k - 1: 2 hashes
    // layer k: final hash --> into merkle proof
    let mut leaf_tree = vec![vec![[F::ZERO; DIGEST_SIZE]; num_leaves]; k + 1];

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
        // i: the index [0, total_depth) within a merkle proof
        let i = if merkle_proof_idx == 0 {
            idx_in_proof
        } else {
            idx_in_proof - num_rows_cum_sums_within_proof[proof_idx][merkle_proof_idx - 1]
        };

        let &MerkleVerifyLog {
            merkle_idx,
            depth,
            query_idx,
            commit_major,
            commit_minor,
        } = &logs_per_proof[proof_idx][merkle_proof_idx];

        if i == 0 {
            if commit_major == 0 {
                for (coset_idx, states) in preflight.initial_row_states[commit_minor][query_idx]
                    .iter()
                    .enumerate()
                {
                    let post_state = states.last().unwrap();
                    leaf_tree[0][coset_idx].copy_from_slice(&post_state[..CHUNK]);
                }
            } else {
                for (coset_idx, state) in preflight.codeword_states[commit_major - 1][query_idx]
                    .iter()
                    .enumerate()
                {
                    leaf_tree[0][coset_idx].copy_from_slice(&state[..CHUNK]);
                }
            }
        }

        let mut stacking_commits = vec![proof.common_main_commit];
        for (air_id, data) in &preflight.proof_shape.sorted_trace_vdata {
            stacking_commits.extend(
                mvk.inner.per_air[*air_id]
                    .preprocessed_data
                    .as_ref()
                    .into_iter()
                    .map(|pdata| pdata.commit)
                    .chain(data.cached_commitments.iter().cloned()),
            );
        }

        let cols: &mut MerkleVerifyCols<F> = row.borrow_mut();
        if idx_in_proof == 0 {
            cols.is_proof_start = F::ONE;
        }
        // determine the layer and offset in the leaf_tree
        // 0th layer: [0, num_leaves / 2)
        // 1st layer: [num_leaves / 2, num_leaves / 2 + num_leaves / 4)
        // kth layer: 1 final hash (that goes into merkle proof)
        let combination_indices = compute_combination_indices(k, i);

        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.merkle_proof_idx = F::from_usize(merkle_proof_idx);
        cols.commit_major = F::from_usize(commit_major);
        cols.commit_minor = F::from_usize(commit_minor);
        cols.total_depth = F::from_usize(depth + k + 1);

        if i == depth + num_leaves - 1 {
            cols.is_last_merkle = F::ONE;
        }
        if i == 0 {
            cols.is_first_merkle = F::ONE;
        }

        if let Some(combination_indices) = combination_indices {
            // combining leaves part
            cols.left =
                leaf_tree[combination_indices.source_layer][combination_indices.left_source_index];
            cols.right =
                leaf_tree[combination_indices.source_layer][combination_indices.right_source_index];
            let output = poseidon2_compress_with_capacity(cols.left, cols.right).0;
            leaf_tree[combination_indices.result_layer][combination_indices.result_index] = output;
            cols.compression_output = output;

            cols.idx = F::from_usize(merkle_idx); // const idx for leaves part
            cols.idx_parity = F::from_usize(merkle_idx % 2);
            cols.height = F::from_usize(combination_indices.source_layer);
            cols.recv_left = F::ONE;
            cols.recv_right = F::ONE;

            let mut input_state = [F::ZERO; POSEIDON2_WIDTH];
            input_state[..DIGEST_SIZE].copy_from_slice(&cols.left);
            input_state[DIGEST_SIZE..].copy_from_slice(&cols.right);
            poseidon2_compress_inputs.push(input_state);

            cols.is_combining_leaves = F::ONE;
            cols.leaf_sub_idx = F::from_usize(combination_indices.result_index);
        } else {
            // merkle proof part
            debug_assert!(i >= num_leaves - 1);
            if i == num_leaves - 1 {
                // The first row of the merkle proof part, initialize cur_hash and cur_idx
                cur_hash = leaf_tree[k][0];
                cur_idx = merkle_idx;
            }
            let pos = i + 1 - num_leaves;
            let is_last = pos == depth;
            let whir_proof = &proof.whir_proof;
            let sibling = match (commit_major, is_last) {
                (0, true) => stacking_commits[commit_minor],
                (0, false) => whir_proof.initial_round_merkle_proofs[commit_minor][query_idx][pos],
                (idx, true) => whir_proof.codeword_commits[idx - 1],
                (idx, false) => whir_proof.codeword_merkle_proofs[idx - 1][query_idx][pos],
            };

            if cur_idx % 2 == 0 {
                cols.left = cur_hash;
                cols.right = sibling;
                cols.recv_left = F::ONE;
            } else {
                cols.left = sibling;
                cols.right = cur_hash;
                cols.recv_right = F::ONE;
            }

            let output = poseidon2_compress_with_capacity(cols.left, cols.right).0;
            cols.compression_output = output;
            let mut input_state = [F::ZERO; POSEIDON2_WIDTH];
            input_state[..DIGEST_SIZE].copy_from_slice(&cols.left);
            input_state[DIGEST_SIZE..].copy_from_slice(&cols.right);
            poseidon2_compress_inputs.push(input_state);

            cols.idx = F::from_usize(cur_idx);
            cols.idx_parity = F::from_usize(cur_idx % 2);
            cols.height = F::from_usize(i + 1 - num_leaves + k);
            cols.is_combining_leaves = F::ZERO;

            cur_hash = output;
            cur_idx /= 2;
        }
    }

    Some((trace, poseidon2_compress_inputs))
}

fn build_merkle_logs(preflight: &Preflight, params: &SystemParams) -> Vec<MerkleVerifyLog> {
    let num_whir_rounds = params.num_whir_rounds();
    let mut logs = Vec::new();
    let mut log_rs_domain_size = params.l_skip + params.n_stack + params.log_blowup;
    let mut query_offset = 0;
    for round_idx in 0..num_whir_rounds {
        let num_queries = params.whir.rounds[round_idx].num_queries;
        for query_idx in 0..num_queries {
            let sample = preflight.whir.queries[query_offset];
            let merkle_idx = sample.as_canonical_u32() as usize;
            let depth = log_rs_domain_size - params.k_whir();

            if round_idx == 0 {
                for commit_minor in 0..preflight.initial_row_states.len() {
                    logs.push(MerkleVerifyLog {
                        merkle_idx,
                        depth,
                        query_idx,
                        commit_major: 0,
                        commit_minor,
                    });
                }
            } else {
                logs.push(MerkleVerifyLog {
                    merkle_idx,
                    depth,
                    query_idx,
                    commit_major: round_idx,
                    commit_minor: 0,
                });
            }

            query_offset += 1;
        }
        log_rs_domain_size -= 1;
    }

    logs
}

// Represents the necessary indices for a single combining operation:
// combining arr[source_layer][left_source_index] and arr[source_layer][right_source_index]
// to produce arr[result_layer][result_index].
#[derive(Debug, PartialEq)]
pub struct CombinationIndices {
    /// The index of the array (vector) containing the two elements to be combined (arr[j]).
    pub source_layer: usize,
    /// The index of the left element in the source layer (arr[j][2*c]).
    pub left_source_index: usize,
    /// The index of the right element in the source layer (arr[j][2*c + 1]).
    pub right_source_index: usize,
    /// The index of the array (vector) where the result is stored (arr[j+1]).
    pub result_layer: usize,
    /// The index of the result element in the result layer (arr[j+1][c]).
    pub result_index: usize,
}

/// Calculates the layer and indices for the i-th combining operation in a complete binary tree
/// with 2^k leaves.
///
/// # Arguments
/// * `k` - The power defining the number of leaves (2^k). Assumed constant for the tree structure.
/// * `i` - The zero-based, overall index of the combining operation (0 <= i < 2^k - 1).
///
/// # Returns
/// An `Option<CombinationIndices>` containing the location of the operation, or `None` if `i` is
/// out of bounds.
pub fn compute_combination_indices(k: usize, i: usize) -> Option<CombinationIndices> {
    if k == 0 {
        // A tree with 2^0 = 1 leaf has no combining operations.
        return None;
    }

    // The total number of combining operations in a complete binary tree is 2^k - 1.
    // 1 << k is equivalent to 2^k.
    let total_operations = (1 << k) - 1;

    if i >= total_operations {
        return None; // Index is out of bounds
    }

    let mut current_index = i;
    let mut source_layer = 0;

    // The number of combining operations at source_layer `j` (which combines elements from
    // arr[j] into arr[j+1]) is 2^(k - (j + 1)).

    // We iterate through layers, subtracting the number of operations in that layer
    // until the `current_index` falls within the range of the current layer.
    while source_layer < k {
        // Calculate C_j = 2^(k - (j + 1))
        let exponent = k - (source_layer + 1);
        let combinations_in_layer = 1 << exponent; // 2^exponent

        if current_index < combinations_in_layer {
            // Found the correct layer!
            let index_within_layer = current_index;

            return Some(CombinationIndices {
                source_layer,
                // The two source elements are always at 2*c and 2*c + 1
                left_source_index: 2 * index_within_layer,
                right_source_index: 2 * index_within_layer + 1,
                // The result is stored in the next layer, at index c
                result_layer: source_layer + 1,
                result_index: index_within_layer,
            });
        }

        // Subtract the count for the current layer and move up to the next layer
        current_index -= combinations_in_layer;
        source_layer += 1;
    }

    // This line should technically be unreachable due to the initial i < total_operations check.
    None
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use itertools::Itertools;
    use openvm_cuda_backend::{base::DeviceMatrix, prelude::F};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};

    use super::*;
    use crate::{
        cuda::{
            preflight::PreflightGpu, proof::ProofGpu, types::MerkleVerifyRecord,
            vk::VerifyingKeyGpu,
        },
        transcript::{cuda_abi, cuda_tracegen::TranscriptBlob},
    };

    const MAX_SUPPORTED_K: usize = 4;

    pub(crate) struct MerkleVerifyBlob {
        pub records: Vec<MerkleVerifyRecord>,
        pub leaf_hashes: Vec<F>,
        pub sibling_hashes: Vec<F>,
        pub proof_row_starts: Vec<usize>,
        pub total_rows: usize,
        pub num_leaves: usize,
        pub k: usize,
        pub num_proofs: usize,
        pub poseidon2_buffer_offset: usize,
    }

    impl MerkleVerifyBlob {
        pub fn new(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            poseidon2_buffer_offset: usize,
        ) -> Self {
            assert_eq!(proofs.len(), preflights.len());
            let k = child_vk.system_params.k_whir();
            assert!(
                k <= MAX_SUPPORTED_K,
                "unsupported k={} for CUDA merkle verify (max {})",
                k,
                MAX_SUPPORTED_K
            );
            let num_leaves = 1 << k;

            let mut records = Vec::new();
            let mut leaf_hashes = Vec::new();
            let mut sibling_hashes = Vec::new();
            let mut proof_row_starts = Vec::with_capacity(proofs.len());

            let stacking_commits_per_proof = proofs
                .iter()
                .zip(preflights)
                .map(|(proof, preflight)| {
                    build_stacking_commits(&child_vk.cpu, &proof.cpu, &preflight.cpu)
                })
                .collect_vec();

            let logs_per_proof: Vec<_> = preflights
                .iter()
                .map(|preflight| build_merkle_logs(&preflight.cpu, &child_vk.system_params))
                .collect();

            let mut total_rows = 0usize;
            for (proof_idx, proof) in proofs.iter().enumerate() {
                let preflight = &preflights[proof_idx].cpu;
                proof_row_starts.push(total_rows);
                for (merkle_proof_idx, log) in logs_per_proof[proof_idx].iter().enumerate() {
                    let num_rows = log.depth + num_leaves;
                    let leaf_offset = leaf_hashes.len();
                    for coset_idx in 0..num_leaves {
                        if log.commit_major == 0 {
                            let states = &preflight.initial_row_states[log.commit_minor]
                                [log.query_idx][coset_idx];
                            let post_state = states.last().unwrap();
                            leaf_hashes.extend_from_slice(&post_state[..CHUNK]);
                        } else {
                            let state = &preflight.codeword_states[log.commit_major - 1]
                                [log.query_idx][coset_idx];
                            leaf_hashes.extend_from_slice(&state[..CHUNK]);
                        }
                    }
                    let path =
                        build_merkle_path(log, &proof.cpu, &stacking_commits_per_proof[proof_idx]);
                    let sibling_offset = sibling_hashes.len();
                    for sibling in path {
                        sibling_hashes.extend_from_slice(&sibling);
                    }
                    records.push(MerkleVerifyRecord {
                        proof_idx: proof_idx as u16,
                        merkle_proof_idx: merkle_proof_idx as u16,
                        start_row: total_rows as u32,
                        num_rows: num_rows as u32,
                        depth: log.depth as u16,
                        merkle_idx: log.merkle_idx as u32,
                        commit_major: log.commit_major as u16,
                        commit_minor: log.commit_minor as u16,
                        leaf_hash_offset: leaf_offset as u32,
                        siblings_offset: sibling_offset as u32,
                    });
                    total_rows += num_rows;
                }
            }

            Self {
                records,
                leaf_hashes,
                sibling_hashes,
                proof_row_starts,
                total_rows,
                num_leaves,
                k,
                num_proofs: proofs.len(),
                poseidon2_buffer_offset,
            }
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub(crate) fn generate_trace(
        blob: &TranscriptBlob,
        required_height: Option<usize>,
    ) -> Option<DeviceMatrix<F>> {
        let merkle_blob = &blob.merkle_verify_blob;
        let trace_width = MerkleVerifyCols::<F>::width();
        let trace_height = if let Some(height) = required_height {
            if height == 0 || merkle_blob.total_rows > height {
                return None;
            }
            height
        } else {
            merkle_blob.total_rows.next_power_of_two().max(1)
        };
        let mut trace = DeviceMatrix::with_capacity(trace_height, trace_width);
        trace.buffer().fill_zero().unwrap();

        if merkle_blob.total_rows == 0 || merkle_blob.records.is_empty() {
            return Some(trace);
        }

        let d_records = merkle_blob.records.to_device().unwrap();
        let d_leaf_hashes = merkle_blob.leaf_hashes.to_device().unwrap();
        let d_siblings = merkle_blob.sibling_hashes.to_device().unwrap();
        let d_proof_row_starts: DeviceBuffer<usize> =
            merkle_blob.proof_row_starts.to_device().unwrap();
        let scratch_len = merkle_blob.records.len() * merkle_blob.num_leaves * DIGEST_SIZE;
        let d_leaf_scratch = DeviceBuffer::<F>::with_capacity(scratch_len);
        d_leaf_scratch.fill_zero().unwrap();

        unsafe {
            cuda_abi::merkle_verify_tracegen(
                &mut trace,
                &d_records,
                &d_leaf_hashes,
                &d_siblings,
                merkle_blob.num_leaves,
                merkle_blob.k,
                &blob.poseidon2_buffer,
                merkle_blob.poseidon2_buffer_offset,
                merkle_blob.total_rows,
                &d_proof_row_starts,
                merkle_blob.num_proofs,
                &d_leaf_scratch,
            )
            .expect("failed to launch merkle verify tracegen");
        }

        Some(trace)
    }

    fn build_stacking_commits(
        mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &Preflight,
    ) -> Vec<[F; DIGEST_SIZE]> {
        let mut commits = vec![proof.common_main_commit];
        for (air_id, data) in &preflight.proof_shape.sorted_trace_vdata {
            if let Some(pdata) = mvk.inner.per_air[*air_id].preprocessed_data.as_ref() {
                commits.push(pdata.commit);
            }
            commits.extend(data.cached_commitments.iter());
        }
        commits
    }

    fn build_merkle_path(
        log: &MerkleVerifyLog,
        proof: &Proof<BabyBearPoseidon2Config>,
        stacking_commits: &[[F; DIGEST_SIZE]],
    ) -> Vec<[F; DIGEST_SIZE]> {
        let whir_proof = &proof.whir_proof;
        (0..=log.depth)
            .map(|pos| {
                let is_last = pos == log.depth;
                match (log.commit_major, is_last) {
                    (0, true) => stacking_commits[log.commit_minor],
                    (0, false) => {
                        whir_proof.initial_round_merkle_proofs[log.commit_minor][log.query_idx][pos]
                    }
                    (idx, true) => whir_proof.codeword_commits[idx - 1],
                    (idx, false) => whir_proof.codeword_merkle_proofs[idx - 1][log.query_idx][pos],
                }
            })
            .collect()
    }
}
