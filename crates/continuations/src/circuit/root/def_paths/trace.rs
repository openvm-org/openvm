use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{instructions::DEFERRAL_AS, POSEIDON2_WIDTH},
    system::memory::dimensions::MemoryDimensions,
};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{prover::AirProvingContext, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    circuit::root::def_paths::air::DeferralAccMerklePathsCols, utils::digests_to_poseidon2_input,
};

fn bit_at(value: usize, bit: usize) -> bool {
    (value >> bit) & 1 == 1
}

fn build_path_nodes(
    acc_hash: [F; DIGEST_SIZE],
    proof: &[[F; DIGEST_SIZE]],
    is_right_child_bits: &[bool],
    skip_depth: usize,
) -> Vec<[F; DIGEST_SIZE]> {
    let num_layers = proof.len() + 1;
    let mut nodes = vec![[F::ZERO; DIGEST_SIZE]; num_layers];

    if skip_depth >= num_layers {
        return nodes;
    }

    nodes[skip_depth] = acc_hash;
    for row_idx in skip_depth..proof.len() {
        let sibling = proof[row_idx];
        let is_right_child = is_right_child_bits[row_idx];
        let left = if is_right_child {
            sibling
        } else {
            nodes[row_idx]
        };
        let right = if is_right_child {
            nodes[row_idx]
        } else {
            sibling
        };
        nodes[row_idx + 1] = poseidon2_compress_with_capacity(left, right).0;
    }

    nodes
}

pub fn generate_proving_input<SC: StarkProtocolConfig<F = F>>(
    initial_acc_hash: [F; DIGEST_SIZE],
    final_acc_hash: [F; DIGEST_SIZE],
    initial_merkle_proof: &[[F; DIGEST_SIZE]],
    final_merkle_proof: &[[F; DIGEST_SIZE]],
    memory_dimensions: MemoryDimensions,
    depth: usize,
    is_unset: bool,
) -> (AirProvingContext<CpuBackend<SC>>, Vec<[F; POSEIDON2_WIDTH]>) {
    assert_eq!(
        initial_merkle_proof.len(),
        final_merkle_proof.len(),
        "initial/final Merkle proofs must have the same depth"
    );

    let proof_len = initial_merkle_proof.len();
    let num_layers = proof_len + 1;
    let height = num_layers.next_power_of_two();
    let width = DeferralAccMerklePathsCols::<u8>::width();

    // Matches AccMerklePathsAir::new().
    let expected_branch_bits = usize::try_from(memory_dimensions.label_to_index((DEFERRAL_AS, 0)))
        .expect("label index must fit in usize");
    let address_height = memory_dimensions.address_height;

    let skip_depth = if is_unset {
        0
    } else {
        assert!(
            depth > 0 && depth <= proof_len,
            "depth must be in 1..=proof_len when is_unset is false"
        );
        depth
    };

    // is_within_deferral_as tracks the DEFERRAL_AS prefix outside the Merkle-rooted part
    let untouched_cut = address_height.min(proof_len.saturating_sub(1));

    let mut is_right_child_bits = vec![false; num_layers];
    let mut row_branch_bits = vec![0usize; num_layers];
    let mut branch_acc = 0usize;
    for row_idx in 0..num_layers {
        let is_right_child = if row_idx < skip_depth {
            false
        } else if is_unset && row_idx == 0 {
            true
        } else {
            bit_at(expected_branch_bits, row_idx)
        };
        is_right_child_bits[row_idx] = is_right_child;
        if is_right_child {
            branch_acc += 1usize << row_idx;
        }
        row_branch_bits[row_idx] = branch_acc;
    }

    let initial_start_hash = if is_unset {
        poseidon2_compress_with_capacity(initial_acc_hash, [F::ZERO; DIGEST_SIZE]).0
    } else {
        initial_acc_hash
    };
    let final_start_hash = if is_unset {
        poseidon2_compress_with_capacity(final_acc_hash, [F::ZERO; DIGEST_SIZE]).0
    } else {
        final_acc_hash
    };

    let initial_nodes = build_path_nodes(
        initial_start_hash,
        initial_merkle_proof,
        &is_right_child_bits,
        skip_depth,
    );
    let final_nodes = build_path_nodes(
        final_start_hash,
        final_merkle_proof,
        &is_right_child_bits,
        skip_depth,
    );
    let final_siblings = final_merkle_proof.to_vec();

    let mut trace = vec![F::ZERO; height * width];
    let mut poseidon2_inputs = Vec::with_capacity(proof_len * 2);

    for row_idx in 0..num_layers {
        let row = &mut trace[row_idx * width..(row_idx + 1) * width];
        let cols: &mut DeferralAccMerklePathsCols<F> = row.borrow_mut();

        cols.is_valid = if row_idx == 0 { F::TWO } else { F::ONE };
        cols.depth = F::from_usize(row_idx);
        cols.is_skip = F::from_bool(row_idx < skip_depth);
        cols.is_within_deferral_as = F::from_bool(row_idx <= untouched_cut);
        cols.is_unset = F::from_bool(is_unset);

        cols.is_right_child = F::from_bool(is_right_child_bits[row_idx]);
        cols.row_idx_exp_2 = F::from_usize(1usize << row_idx);
        cols.merkle_path_branch_bits = F::from_usize(row_branch_bits[row_idx]);

        cols.initial_node_commit = initial_nodes[row_idx];
        cols.final_node_commit = final_nodes[row_idx];
        if row_idx < proof_len {
            cols.initial_sibling = initial_merkle_proof[row_idx];
            cols.final_sibling = final_siblings[row_idx];
        }
    }

    for row_idx in 0..proof_len {
        if row_idx < skip_depth {
            continue;
        }
        let is_right_child = is_right_child_bits[row_idx];

        let init_left = if is_right_child {
            initial_merkle_proof[row_idx]
        } else {
            initial_nodes[row_idx]
        };
        let init_right = if is_right_child {
            initial_nodes[row_idx]
        } else {
            initial_merkle_proof[row_idx]
        };
        poseidon2_inputs.push(digests_to_poseidon2_input(init_left, init_right));

        let final_left = if is_right_child {
            final_siblings[row_idx]
        } else {
            final_nodes[row_idx]
        };
        let final_right = if is_right_child {
            final_nodes[row_idx]
        } else {
            final_siblings[row_idx]
        };
        poseidon2_inputs.push(digests_to_poseidon2_input(final_left, final_right));
    }

    (
        AirProvingContext::simple_no_pis(RowMajorMatrix::new(trace, width)),
        poseidon2_inputs,
    )
}
