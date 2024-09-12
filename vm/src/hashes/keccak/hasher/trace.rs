use std::array::from_fn;

use afs_stark_backend::rap::AnyRap;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_keccak_air::{
    generate_trace_rows, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, NUM_ROUNDS, U64_LIMBS,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig};
use tiny_keccak::keccakf;

use super::{KeccakVmChip, KECCAK_DIGEST_WRITES};
use crate::{
    arch::chips::MachineChip,
    hashes::keccak::hasher::{
        columns::{KeccakOpcodeCols, KeccakVmColsMut},
        KECCAK_ABSORB_READS, KECCAK_EXECUTION_READS, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
    },
    memory::{offline_checker::MemoryReadAuxCols, MemoryReadRecord, MemoryWriteRecord},
};

impl<F: PrimeField32> MachineChip<F> for KeccakVmChip<F> {
    /// This should only be called once. It takes all records from the chip state.
    fn generate_trace(mut self) -> RowMajorMatrix<F> {
        let records = std::mem::take(&mut self.records);
        let total_num_blocks: usize = records.iter().map(|r| r.input_blocks.len()).sum();
        let mut states = Vec::with_capacity(total_num_blocks);
        let mut opcode_blocks = Vec::with_capacity(total_num_blocks);

        #[derive(Clone)]
        struct StateDiff<F> {
            /// hi-byte of pre-state
            pre_hi: [u8; KECCAK_RATE_U16S],
            /// hi-byte of post-state
            post_hi: [u8; KECCAK_RATE_U16S],
            /// if first block
            op_reads: Option<[MemoryReadRecord<F, 1>; 3]>,
            /// if last block
            digest_writes: Option<[MemoryWriteRecord<F, 1>; KECCAK_DIGEST_WRITES]>,
        }

        impl<F> Default for StateDiff<F> {
            fn default() -> Self {
                Self {
                    pre_hi: [0; KECCAK_RATE_U16S],
                    post_hi: [0; KECCAK_RATE_U16S],
                    op_reads: None,
                    digest_writes: None,
                }
            }
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
            let num_blocks = record.input_blocks.len();
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
                let op_reads = (idx == 0).then(|| {
                    [
                        record.dst_read.clone(),
                        record.src_read.clone(),
                        record.len_read.clone(),
                    ]
                });
                let digest_writes = (idx == num_blocks - 1).then(|| record.digest_writes.clone());
                let diff = StateDiff {
                    pre_hi,
                    post_hi,
                    op_reads,
                    digest_writes,
                };
                opcode_blocks.push((opcode, diff, block));
                opcode.len -= F::from_canonical_usize(KECCAK_RATE_BYTES);
                opcode.src += F::from_canonical_usize(KECCAK_RATE_BYTES);
                opcode.start_timestamp +=
                    F::from_canonical_usize(KECCAK_EXECUTION_READS + KECCAK_ABSORB_READS);
            }
        }

        let p3_keccak_trace: RowMajorMatrix<F> = generate_trace_rows(states);
        let num_rows = p3_keccak_trace.height();
        // Every `NUM_ROUNDS` rows corresponds to one input block
        let num_blocks = (num_rows + NUM_ROUNDS - 1) / NUM_ROUNDS;
        // Resize with dummy `is_opcode = 0`
        opcode_blocks.resize(num_blocks, Default::default());

        let memory = self.memory_chip.borrow();
        let mem_oc = memory.make_offline_checker();
        let range_checker = memory.range_checker.clone();
        drop(memory);
        // Use unsafe alignment so we can parallely write to the matrix
        let trace_width = self.trace_width();
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
            .for_each(|((rows, p3_keccak_mat), (opcode, diff, block))| {
                let height = rows.len() / trace_width;
                for (row, p3_keccak_row) in rows
                    .chunks_exact_mut(trace_width)
                    .zip(p3_keccak_mat.chunks_exact(NUM_KECCAK_PERM_COLS))
                {
                    // Safety: `KeccakPermCols` **must** be the first field in `KeccakVmCols`
                    row[..NUM_KECCAK_PERM_COLS].copy_from_slice(p3_keccak_row);
                    let row_mut = KeccakVmColsMut::from_mut_slice(row);
                    *row_mut.opcode = opcode;

                    row_mut.sponge.block_bytes = block.padded_bytes.map(F::from_canonical_u8);
                    for (i, is_padding) in row_mut.sponge.is_padding_byte.iter_mut().enumerate() {
                        *is_padding = F::from_bool(i >= block.remaining_len);
                    }
                }
                let first_row = KeccakVmColsMut::from_mut_slice(&mut rows[..trace_width]);
                first_row.sponge.is_new_start = F::from_bool(block.is_new_start);
                first_row.sponge.state_hi = diff.pre_hi.map(F::from_canonical_u8);
                // Make memory access aux columns. Any aux column not explicitly defined defaults to all 0s
                let mut slc_idx = 0;
                if let Some(op_reads) = diff.op_reads {
                    for record in op_reads {
                        let aux = mem_oc
                            .make_read_aux_cols(range_checker.clone(), record)
                            .flatten();
                        first_row.mem_oc[slc_idx..][..aux.len()].copy_from_slice(&aux);
                        slc_idx += aux.len();
                    }
                }
                slc_idx = KECCAK_EXECUTION_READS * MemoryReadAuxCols::<F, 1>::width();
                for record in block.bytes_read {
                    // TODO[jpw] make_read_aux_cols should directly write into slice
                    let aux = mem_oc
                        .make_read_aux_cols(range_checker.clone(), record)
                        .flatten();
                    first_row.mem_oc[slc_idx..][..aux.len()].copy_from_slice(&aux);
                    slc_idx += aux.len();
                }

                let last_row =
                    KeccakVmColsMut::from_mut_slice(&mut rows[(height - 1) * trace_width..]);
                last_row.sponge.state_hi = diff.post_hi.map(F::from_canonical_u8);
                last_row.inner.export =
                    opcode.is_enabled * F::from_bool(block.remaining_len < KECCAK_RATE_BYTES);
                if let Some(digest_writes) = diff.digest_writes {
                    slc_idx = (KECCAK_EXECUTION_READS + KECCAK_ABSORB_READS)
                        * MemoryReadAuxCols::<F, 1>::width();
                    for record in digest_writes {
                        // TODO: these aux columns are only used for the last row - can we share them with aux reads in first row?
                        let aux = mem_oc
                            .make_write_aux_cols(range_checker.clone(), record)
                            .flatten();
                        last_row.mem_oc[slc_idx..][..aux.len()].copy_from_slice(&aux);
                        slc_idx += aux.len();
                    }
                }
            });

        trace
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let num_blocks: usize = self.records.iter().map(|r| r.input_blocks.len()).sum();
        num_blocks * NUM_ROUNDS
    }
}
