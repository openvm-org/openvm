use std::array::from_fn;

use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_keccak_air::{
    generate_trace_rows, KeccakCols as KeccakPermCols, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS,
    NUM_ROUNDS, U64_LIMBS,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use tiny_keccak::keccakf;

use super::{KeccakVmChip, KECCAK_DIGEST_WRITES};
use crate::{
    arch::chips::MachineChip,
    hashes::keccak::hasher::{
        columns::{KeccakMemoryCols, KeccakOpcodeCols, KeccakVmColsMut},
        KECCAK_ABSORB_READS, KECCAK_EXECUTION_READS, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
    },
    memory::{
        manager::{MemoryAccess, MemoryChip},
        offline_checker::columns::MemoryWriteAuxCols,
    },
};

impl<F: PrimeField32> MachineChip<F> for KeccakVmChip<F> {
    /// This should only be called once. It takes all records from the chip state.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let records = std::mem::take(&mut self.records);
        let total_num_blocks: usize = records.iter().map(|r| r.input_blocks.len()).sum();
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
        struct AuxBlock<F>  {
            diff: StateDiff,
            op_reads: [MemoryAccess<1,F>; 3],
            digest_writes: [MemoryAccess<1,F>; KECCAK_DIGEST_WRITES],
        }

        // prepare the states
        let mut state: [u64; 25];
        for record in records {
            state = [0u64; 25];
            let [a, b, c, d, e, f] = record.operands();
            let mut opcode = KeccakOpcodeCols {
                pc: record.pc,
                is_enabled: F::one(),
                start_timestamp: record.start_timestamp(),
                a,
                b,
                c,
                d,
                e,
                f,
                dst: record.dst(),
                src: record.src(),
                len: record.len(),
            };
            for (idx, block) in record.input_blocks.into_iter().enumerate() {
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
                opcode.start_timestamp += F::from_canonical_usize(KECCAK_EXECUTION_READS + KECCAK_ABSORB_READS);
            }
        }

        let p3_keccak_trace: RowMajorMatrix<F> = generate_trace_rows(states);
        let num_rows = p3_keccak_trace.height();
        // Every `NUM_ROUNDS` rows corresponds to one input block
        let num_blocks = (num_rows + NUM_ROUNDS - 1) / NUM_ROUNDS;
        // Resize with dummy `is_opcode = 0`
        opcode_blocks.resize(num_blocks, Default::default());

        let memory = self.memory_chip.borrow();
        // Use unsafe alignment so we can parallely write to the matrix
        let trace_width = self.air.width();
        let mut trace = RowMajorMatrix::new(vec![F::zero(); num_rows * trace_width], trace_width);

        trace
            .values
            .par_chunks_mut(trace_width * NUM_ROUNDS)
            .zip(
                p3_keccak_trace
                    .values
                    .par_chunks(NUM_KECCAK_PERM_COLS * NUM_ROUNDS),
            )
            .zip(opcode_blocks.into_par_iter())
            .for_each(|((rows, p3_keccak_mat), (opcode, diff, mut block))| {
                let height = rows.len() / trace_width;
                for (row_idx,(row, p3_keccak_row)) in rows
                    .chunks_exact_mut(trace_width)
                    .zip(p3_keccak_mat.chunks_exact(NUM_KECCAK_PERM_COLS)).enumerate()
                {
                    // Safety: `KeccakPermCols` **must** be the first field in `KeccakVmCols`
                    row[..NUM_KECCAK_PERM_COLS].copy_from_slice(p3_keccak_row);
                    let mut row = KeccakVmColsMut::from_mut_slice(row);
                    *row.opcode = opcode;

                    row.sponge.block_bytes = block.padded_bytes.map(F::from_canonical_u8);
                    for (i, is_padding) in row.sponge.is_padding_byte.iter_mut().enumerate() {
                        *is_padding = F::from_bool(i >= block.remaining_len);
                    }

                    // Extend bytes_read with dummy reads to fixed length
                    let absorb_reads = from_fn(|i| {
                        if let Some(read) = block.bytes_read.get(i) {
                            memory.make_access_cols(*read)
                        } else {
                            memory.make_access_cols(MemoryAccess::disabled_read(
                                block.start_read_timestamp + F::from_canonical_usize(i),
                                opcode.e,
                            ))
                        }
                    });
                    let start_write_timestamp =
                        block.start_read_timestamp + F::from_canonical_usize(KECCAK_ABSORB_READS);

                    let op_reads =
                    let mem = KeccakMemoryCols {
                        // Disabled. Only first row will have real read.
                        // We don't advance timestamp for dummy reads
                        op_reads: [opcode.d, opcode.d, opcode.f].map(|address_space| {
                            memory.make_access_cols(MemoryAccess::disabled_read(
                                opcode.start_timestamp,
                                address_space,
                            ))
                        }),
                        absorb_reads,
                        digest_writes: disabled_write_chunk(
                            &memory,
                            start_write_timestamp,
                            opcode.e,
                        ),
                    };
                    row.mem_oc.copy_from_slice(&mem.flatten());
                }
                let first_row = KeccakVmColsMut::from_mut_slice(&mut rows[..trace_width]);
                first_row.sponge.is_new_start = F::from_bool(block.is_new_start);
                first_row.sponge.state_hi = diff.pre_hi.map(F::from_canonical_u8);
                let last_row =
                    KeccakVmColsMut::from_mut_slice(&mut rows[(height - 1) * trace_width..]);
                last_row.sponge.state_hi = diff.post_hi.map(F::from_canonical_u8);
                last_row.inner.export =
                    opcode.is_enabled * F::from_bool(block.remaining_len < KECCAK_RATE_BYTES);
            });

        trace
    }
}

fn disabled_write_cols<F: PrimeField32, const N: usize>(
    memory: &MemoryChip<F>,
    start_timestamp: F,
    address_space: F,
) -> [MemoryWriteAuxCols<1, F>; N] {
    from_fn(|i| {
        memory.make_access_cols(MemoryAccess::disabled_write(
            start_timestamp + F::from_canonical_usize(i),
            address_space,
        ))
    })
}
