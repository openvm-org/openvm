use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{
        KeccakSpongeCols, KECCAK_DIGEST_U16S, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
        KECCAK_WIDTH_U16S, NUM_KECCAK_SPONGE_COLS,
    },
    util::keccakf_u16s,
    KeccakSpongeAir, KeccakSpongeOp,
};

impl KeccakSpongeAir {
    pub fn generate_trace<F: PrimeField32>(
        &self,
        inputs: Vec<KeccakSpongeOp>,
    ) -> RowMajorMatrix<F> {
        let num_cols = NUM_KECCAK_SPONGE_COLS;
        // Generate the witness row-wise.
        let num_rows = inputs
            .iter()
            .map(|op| op.input.len() / KECCAK_RATE_BYTES + 1)
            .sum::<usize>()
            .next_power_of_two();
        let mut trace = RowMajorMatrix::new(vec![F::zero(); num_rows * num_cols], num_cols);
        let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakSpongeCols<F>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), num_rows);

        generate_trace_rows(rows, inputs.as_slice());

        trace
    }
}

pub fn generate_trace_rows<F: PrimeField32>(
    rows: &mut [KeccakSpongeCols<F>],
    inputs: &[KeccakSpongeOp],
) {
    let mut offset = 0;
    for op in inputs.iter() {
        let len = op.input.len() / KECCAK_RATE_BYTES + 1;
        let input_rows = &mut rows[offset..offset + len];
        generate_rows_for_op(input_rows, op);
        offset += len;
    }

    // Pad the trace.
    for input_rows in rows.chunks_mut(1).skip(offset) {
        generate_rows_for_op(input_rows, &KeccakSpongeOp::default());
    }
}

/// Generates the rows associated to a given operation:
/// Performs a Keccak sponge permutation and fills the STARK's rows
/// accordingly. The number of rows is the number of input chunks of
/// size `KECCAK_RATE_BYTES`.
fn generate_rows_for_op<F: PrimeField32>(rows: &mut [KeccakSpongeCols<F>], op: &KeccakSpongeOp) {
    let mut sponge_state = [0u16; KECCAK_WIDTH_U16S];

    let KeccakSpongeOp {
        addr: _,
        timestamp: _,
        input,
    } = op;

    let mut input_blocks = input.chunks_exact(KECCAK_RATE_BYTES);
    let mut already_absorbed_bytes = 0;
    for (row, block) in rows.iter_mut().zip(input_blocks.by_ref()) {
        // We compute the updated state of the sponge.
        generate_full_input_row::<F>(
            row,
            op,
            already_absorbed_bytes,
            sponge_state,
            block.try_into().unwrap(),
        );

        // We update the state limbs for the next block absorption.
        // The first `KECCAK_DIGEST_U16s` limbs are stored as bytes after the
        // computation, so we recompute the corresponding `u16` and update
        // the first state limbs.
        sponge_state[..KECCAK_DIGEST_U16S]
            .iter_mut()
            .zip(row.updated_digest_state_bytes.chunks_exact(2))
            .for_each(|(s, bs)| {
                *s = bs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (b.as_canonical_u64() as u16) << (8 * i))
                    .sum();
            });

        // The rest of the bytes are already stored in the expected form, so we can
        // directly update the state with the stored values.
        sponge_state[KECCAK_DIGEST_U16S..]
            .iter_mut()
            .zip(row.partial_updated_state_u16s)
            .for_each(|(s, x)| *s = x.as_canonical_u64() as u16);

        already_absorbed_bytes += KECCAK_RATE_BYTES;
    }

    generate_final_row(
        rows.last_mut().unwrap(),
        op,
        already_absorbed_bytes,
        sponge_state,
        input_blocks.remainder(),
    );
}

/// Generates a row where all bytes are input bytes, not padding bytes.
/// This includes updating the state sponge with a single absorption.
fn generate_full_input_row<F: PrimeField32>(
    row: &mut KeccakSpongeCols<F>,
    op: &KeccakSpongeOp,
    already_absorbed_bytes: usize,
    sponge_state: [u16; KECCAK_WIDTH_U16S],
    block: [u8; KECCAK_RATE_BYTES],
) {
    row.is_full_input_block = F::one();
    row.block_bytes = block.map(F::from_canonical_u8);

    generate_common_fields(row, op, already_absorbed_bytes, sponge_state);
}

/// Generates a row containing the last input bytes.
fn generate_final_row<F: PrimeField32>(
    row: &mut KeccakSpongeCols<F>,
    op: &KeccakSpongeOp,
    already_absorbed_bytes: usize,
    sponge_state: [u16; KECCAK_WIDTH_U16S],
    final_inputs: &[u8],
) {
    assert_eq!(already_absorbed_bytes + final_inputs.len(), op.input.len());

    for (block_byte, input_byte) in row.block_bytes.iter_mut().zip(final_inputs) {
        *block_byte = F::from_canonical_u8(*input_byte);
    }

    // pad10*1 rule
    if final_inputs.len() == KECCAK_RATE_BYTES - 1 {
        // Both 1s are placed in the same byte.
        row.block_bytes[final_inputs.len()] = F::from_canonical_u8(0b10000001);
    } else {
        row.block_bytes[final_inputs.len()] = F::one();
        row.block_bytes[KECCAK_RATE_BYTES - 1] = F::from_canonical_u8(0b10000000);
    }

    for i in final_inputs.len()..KECCAK_RATE_BYTES {
        row.is_padding_byte[i] = F::one();
    }

    generate_common_fields(row, op, already_absorbed_bytes, sponge_state)
}

/// Generate fields that are common to both full-input-block rows and
/// final-block rows. Also updates the sponge state with a single
/// absorption. Given a state S = R || C and a block input B,
/// - R is updated with R XOR B,
/// - S is replaced by keccakf_u16s(S).
fn generate_common_fields<F: PrimeField32>(
    row: &mut KeccakSpongeCols<F>,
    op: &KeccakSpongeOp,
    already_absorbed_bytes: usize,
    mut sponge_state: [u16; KECCAK_WIDTH_U16S],
) {
    row.timestamp = F::from_canonical_u32(op.timestamp);
    row.base_addr = F::from_canonical_u32(op.addr);
    row.already_absorbed_bytes = F::from_canonical_usize(already_absorbed_bytes);

    row.original_rate_u16s = sponge_state[..KECCAK_RATE_U16S]
        .iter()
        .map(|x| F::from_canonical_u16(*x))
        .collect_vec()
        .try_into()
        .unwrap();

    row.original_capacity_u16s = sponge_state[KECCAK_RATE_U16S..]
        .iter()
        .map(|x| F::from_canonical_u16(*x))
        .collect_vec()
        .try_into()
        .unwrap();

    let block_u16s = (0..KECCAK_RATE_U16S).map(|i| {
        u16::from_le_bytes(
            row.block_bytes[i * 2..(i + 1) * 2]
                .iter()
                .map(|x| x.as_canonical_u64() as u8)
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    });

    // xor in the block
    for (state_i, block_i) in sponge_state.iter_mut().zip(block_u16s) {
        *state_i ^= block_i;
    }
    let xored_rate_u16s: [u16; KECCAK_RATE_U16S] = sponge_state[..KECCAK_RATE_U16S]
        .to_vec()
        .try_into()
        .unwrap();
    row.xored_rate_u16s = xored_rate_u16s.map(F::from_canonical_u16);

    keccakf_u16s(&mut sponge_state);
    // Store all but the first `KECCAK_DIGEST_U16S` limbs in the updated state.
    // Those missing limbs will be broken down into bytes and stored separately.
    row.partial_updated_state_u16s.copy_from_slice(
        &sponge_state[KECCAK_DIGEST_U16S..]
            .iter()
            .copied()
            .map(|i| F::from_canonical_u16(i))
            .collect_vec(),
    );
    sponge_state[..KECCAK_DIGEST_U16S]
        .iter()
        .enumerate()
        .for_each(|(l, &elt)| {
            let mut cur_elt = elt;
            (0..2).for_each(|i| {
                row.updated_digest_state_bytes[l * 2 + i] = F::from_canonical_u16(cur_elt & 0xFF);
                cur_elt >>= 8;
            });

            // 16-bit limb reconstruction consistency check.
            let mut s = row.updated_digest_state_bytes[l * 2].as_canonical_u64();
            for i in 1..2 {
                s += row.updated_digest_state_bytes[l * 2 + i].as_canonical_u64() << (8 * i);
            }
            assert_eq!(elt as u64, s, "not equal");
        })
}
