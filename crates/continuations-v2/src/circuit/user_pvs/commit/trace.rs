use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, CpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::{
    root::digests_to_poseidon2_input,
    user_pvs::commit::{UserPvsCommitCols, MAX_ENCODER_DEGREE},
};

pub fn generate_proving_ctx(
    user_pvs: Vec<F>,
    expose_public_values: bool,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    let num_user_pvs = user_pvs.len();

    // Each leaf consumes `DIGEST_SIZE` public values, which is padded and hashed before
    // being inserted into the Merkle tree. We require at least one leaf (so at least one
    // Poseidon2 hash), and a full binary tree.
    debug_assert!(num_user_pvs >= DIGEST_SIZE);
    debug_assert!(num_user_pvs.is_multiple_of(DIGEST_SIZE));
    debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());

    // One selector per leaf PV chunk (each leaf consumes 1 digest).
    let encoder = expose_public_values.then_some(Encoder::new(
        num_user_pvs / DIGEST_SIZE,
        MAX_ENCODER_DEGREE,
        true,
    ));

    let num_pv_digests = num_user_pvs / DIGEST_SIZE;
    let const_width = UserPvsCommitCols::<u8>::width();
    let width = const_width + encoder.as_ref().map_or(0, |e| e.width());
    let mut trace = vec![F::ZERO; 2 * num_pv_digests * width];
    let mut chunks = trace.chunks_mut(width);

    let mut next_layer = Vec::with_capacity(num_pv_digests);
    let mut poseidon2_compress_inputs = Vec::with_capacity(2 * num_pv_digests - 1);

    // Write leaf nodes that read each digest child from pvs
    for pv_digest in user_pvs.chunks(DIGEST_SIZE) {
        let chunk = chunks.next().unwrap();
        let cols: &mut UserPvsCommitCols<F> = chunk[..const_width].borrow_mut();

        let row_idx = next_layer.len();
        let left: [F; DIGEST_SIZE] = pv_digest[..DIGEST_SIZE].try_into().unwrap();
        let right: [F; DIGEST_SIZE] = [F::ZERO; DIGEST_SIZE];
        let parent = poseidon2_compress_with_capacity(left, right).0;
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));

        cols.row_idx = F::from_usize(row_idx);
        cols.send_type = if num_pv_digests == 1 { F::TWO } else { F::ONE };
        cols.receive_type = F::ONE;
        cols.parent = parent;
        cols.is_right_child = F::from_bool(row_idx & 1 == 1);
        cols.left_child = left;
        cols.right_child = right;

        if let Some(encoder) = &encoder {
            chunk[const_width..].copy_from_slice(
                encoder
                    .get_flag_pt(row_idx)
                    .into_iter()
                    .map(F::from_u32)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
        }
        next_layer.push(parent);
    }

    let mut row_idx = next_layer.len();

    // Write internal nodes that receive left and right child from bus
    while next_layer.len() > 1 {
        let parent_layer_len = next_layer.len() >> 1;
        for parent_idx in 0..parent_layer_len {
            let chunk = chunks.next().unwrap();
            let cols: &mut UserPvsCommitCols<F> = chunk[..const_width].borrow_mut();

            let left = next_layer[2 * parent_idx];
            let right = next_layer[2 * parent_idx + 1];
            let parent = poseidon2_compress_with_capacity(left, right).0;
            poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));

            cols.row_idx = F::from_usize(row_idx);
            cols.send_type = if parent_layer_len > 1 { F::ONE } else { F::TWO };
            cols.receive_type = F::TWO;
            cols.parent = parent;
            cols.is_right_child = F::from_bool(row_idx & 1 == 1);
            cols.left_child = left;
            cols.right_child = right;

            row_idx += 1;
            next_layer[parent_idx] = parent;
        }
        next_layer.truncate(parent_layer_len);
    }

    debug_assert_eq!(row_idx + 1, 2 * num_pv_digests);
    let last_chunk = chunks.next().unwrap();
    let last_cols: &mut UserPvsCommitCols<F> = last_chunk[..const_width].borrow_mut();
    last_cols.row_idx = F::from_usize(row_idx);
    last_cols.is_right_child = F::ONE;

    let common_main = ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width));
    let ctx = if expose_public_values {
        AirProvingContext::simple(common_main, user_pvs)
    } else {
        AirProvingContext::simple_no_pis(common_main)
    };
    (ctx, poseidon2_compress_inputs)
}
