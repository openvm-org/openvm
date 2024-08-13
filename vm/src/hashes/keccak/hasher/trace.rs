use std::array::from_fn;

use p3_field::PrimeField32;
use p3_keccak_air::{generate_trace_rows, KeccakCols, NUM_KECCAK_COLS, NUM_ROUNDS, U64_LIMBS};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use tiny_keccak::keccakf;

use super::KeccakVmChip;
use crate::hashes::keccak::hasher::{
    bridge::{BLOCK_MEMORY_ACCESSES, TIMESTAMP_OFFSET_FOR_OPCODE},
    columns::{KeccakVmCols, NUM_KECCAK_VM_COLS},
    KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
};

impl<F: PrimeField32> KeccakVmChip<F> {
    pub fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let requests = std::mem::take(&mut self.requests);
        let total_num_blocks: usize = requests.iter().map(|(_, block)| block.len()).sum();
        let mut states = Vec::with_capacity(total_num_blocks);
        let mut opcode_blocks = Vec::with_capacity(total_num_blocks);

        #[derive(Clone)]
        struct StateDiff {
            /// hi-byte of pre-state
            pre_hi: [u8; KECCAK_RATE_U16S],
            /// hi-byte of post-state
            post_hi: [u8; KECCAK_RATE_U16S],
        }
        impl Default for StateDiff {
            fn default() -> Self {
                Self {
                    pre_hi: [0; KECCAK_RATE_U16S],
                    post_hi: [0; KECCAK_RATE_U16S],
                }
            }
        }

        // prepare the states
        let mut state: [u64; 25];
        for (mut opcode, blocks) in requests {
            state = [0u64; 25];
            for block in blocks {
                // absorb
                for (bytes, s) in block.padded_bytes.chunks_exact(8).zip(state.iter_mut()) {
                    // u64 <-> bytes conversion is little-endian
                    for (i, &byte) in bytes.iter().enumerate() {
                        let s_byte = (*s >> (i * 8)) as u8;
                        // Update xor chip state: order matters!
                        self.byte_xor_chip.request(byte as u32, s_byte as u32);
                        *s ^= (byte as u64) << (i * 8);
                    }
                }
                let pre_hi: [u8; KECCAK_RATE_U16S] =
                    from_fn(|i| (state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)) as u8);
                states.push(state);
                keccakf(&mut state);
                let post_hi: [u8; KECCAK_RATE_U16S] =
                    from_fn(|i| (state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)) as u8);
                let diff = StateDiff { pre_hi, post_hi };
                opcode_blocks.push((opcode, diff, block));
                opcode.len -= F::from_canonical_usize(KECCAK_RATE_BYTES);
                opcode.src += F::from_canonical_usize(KECCAK_RATE_BYTES);
                opcode.start_timestamp +=
                    F::from_canonical_usize(TIMESTAMP_OFFSET_FOR_OPCODE + BLOCK_MEMORY_ACCESSES);
            }
        }

        let p3_keccak_trace: RowMajorMatrix<F> = generate_trace_rows(states);
        let num_rows = p3_keccak_trace.height();
        // Every `NUM_ROUNDS` rows corresponds to one input block
        let num_blocks = (num_rows + NUM_ROUNDS - 1) / NUM_ROUNDS;
        // Resize with dummy `is_opcode = 0`
        opcode_blocks.resize(num_blocks, Default::default());

        // Use unsafe alignment so we can parallely write to the matrix
        let mut trace = RowMajorMatrix::new(
            vec![F::zero(); num_rows * NUM_KECCAK_VM_COLS],
            NUM_KECCAK_VM_COLS,
        );
        let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakVmCols<F>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), num_rows);

        rows.par_chunks_mut(NUM_ROUNDS)
            .zip(
                p3_keccak_trace
                    .values
                    .par_chunks(NUM_KECCAK_COLS * NUM_ROUNDS),
            )
            .zip(opcode_blocks.into_par_iter())
            .for_each(|((rows, p3_keccak_mat), (opcode, diff, block))| {
                for (row, p3_keccak_row) in rows
                    .iter_mut()
                    .zip(p3_keccak_mat.chunks_exact(NUM_KECCAK_COLS))
                {
                    // Cast &mut KeccakCols<F> to &mut [F]:
                    let inner_raw_ptr: *mut KeccakCols<F> = &mut row.inner as *mut _;
                    // Safety: `KeccakPermCols` **must** be the first field in `KeccakVmCols`
                    let row_slice = unsafe {
                        std::slice::from_raw_parts_mut(inner_raw_ptr as *mut F, NUM_KECCAK_COLS)
                    };
                    row_slice.copy_from_slice(p3_keccak_row);

                    row.opcode = opcode;

                    row.sponge.block_bytes = block.padded_bytes.map(F::from_canonical_u8);
                    for (i, is_padding) in row.sponge.is_padding_byte.iter_mut().enumerate() {
                        *is_padding = F::from_bool(i >= block.remaining_len);
                    }
                }
                rows[0].sponge.is_new_start = F::from_bool(block.is_new_start);
                rows[0].sponge.state_hi = diff.pre_hi.map(F::from_canonical_u8);
                let last_row = rows.last_mut().unwrap();
                last_row.sponge.state_hi = diff.post_hi.map(F::from_canonical_u8);
                last_row.inner.export =
                    opcode.is_enabled * F::from_bool(block.remaining_len < KECCAK_RATE_BYTES);
            });

        trace
    }
}
