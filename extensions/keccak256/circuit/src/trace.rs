use std::{array::from_fn, borrow::BorrowMut, sync::Arc};

use openvm_circuit::system::memory::RecordId;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::get_air_name,
    AirRef, Chip, ChipUsageGetter,
};
use p3_keccak_air::{
    generate_trace_rows, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, NUM_ROUNDS, U64_LIMBS,
};
use tiny_keccak::keccakf;

use super::{
    columns::{KeccakInstructionCols, KeccakVmCols},
    KeccakVmChip, KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
    KECCAK_REGISTER_READS, NUM_ABSORB_ROUNDS,
};

impl<SC: StarkGenericConfig> Chip<SC> for KeccakVmChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace_width = self.trace_width();
        let records = self.records;
        let total_num_blocks: usize = records.iter().map(|r| r.input_blocks.len()).sum();
        let mut states = Vec::with_capacity(total_num_blocks);
        let mut instruction_blocks = Vec::with_capacity(total_num_blocks);
        let memory = self.offline_memory.lock().unwrap();

        #[derive(Clone)]
        struct StateDiff {
            /// hi-byte of pre-state
            pre_hi: [u8; KECCAK_RATE_U16S],
            /// hi-byte of post-state
            post_hi: [u8; KECCAK_RATE_U16S],
            /// if first block
            register_reads: Option<[RecordId; KECCAK_REGISTER_READS]>,
            /// if last block
            digest_writes: Option<[RecordId; KECCAK_DIGEST_WRITES]>,
        }

        impl Default for StateDiff {
            fn default() -> Self {
                Self {
                    pre_hi: [0; KECCAK_RATE_U16S],
                    post_hi: [0; KECCAK_RATE_U16S],
                    register_reads: None,
                    digest_writes: None,
                }
            }
        }

        // prepare the states
        let mut state: [u64; 25];
        for record in records {
            let dst_read = memory.record_by_id(record.dst_read);
            let src_read = memory.record_by_id(record.src_read);
            let len_read = memory.record_by_id(record.len_read);

            state = [0u64; 25];
            let src_limbs: [_; RV32_REGISTER_NUM_LIMBS - 1] = src_read.data_slice()
                [1..RV32_REGISTER_NUM_LIMBS]
                .try_into()
                .unwrap();
            let len_limbs: [_; RV32_REGISTER_NUM_LIMBS - 1] = len_read.data_slice()
                [1..RV32_REGISTER_NUM_LIMBS]
                .try_into()
                .unwrap();
            let mut instruction = KeccakInstructionCols {
                pc: record.pc,
                is_enabled: Val::<SC>::ONE,
                is_enabled_first_round: Val::<SC>::ZERO,
                start_timestamp: Val::<SC>::from_canonical_u32(dst_read.timestamp),
                dst_ptr: dst_read.pointer,
                src_ptr: src_read.pointer,
                len_ptr: len_read.pointer,
                dst: dst_read.data_slice().try_into().unwrap(),
                src_limbs,
                src: Val::<SC>::from_canonical_usize(record.input_blocks[0].src),
                len_limbs,
                remaining_len: Val::<SC>::from_canonical_usize(
                    record.input_blocks[0].remaining_len,
                ),
            };
            let num_blocks = record.input_blocks.len();
            for (idx, block) in record.input_blocks.into_iter().enumerate() {
                // absorb
                for (bytes, s) in block.padded_bytes.chunks_exact(8).zip(state.iter_mut()) {
                    // u64 <-> bytes conversion is little-endian
                    for (i, &byte) in bytes.iter().enumerate() {
                        let s_byte = (*s >> (i * 8)) as u8;
                        // Update bitwise lookup (i.e. xor) chip state: order matters!
                        if idx != 0 {
                            self.bitwise_lookup_chip
                                .request_xor(byte as u32, s_byte as u32);
                        }
                        *s ^= (byte as u64) << (i * 8);
                    }
                }
                let pre_hi: [u8; KECCAK_RATE_U16S] =
                    from_fn(|i| (state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)) as u8);
                states.push(state);
                keccakf(&mut state);
                let post_hi: [u8; KECCAK_RATE_U16S] =
                    from_fn(|i| (state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)) as u8);
                // Range check the final state
                if idx == num_blocks - 1 {
                    for s in state.into_iter().take(NUM_ABSORB_ROUNDS) {
                        for s_byte in s.to_le_bytes() {
                            self.bitwise_lookup_chip.request_xor(0, s_byte as u32);
                        }
                    }
                }
                let register_reads =
                    (idx == 0).then_some([record.dst_read, record.src_read, record.len_read]);
                let digest_writes = (idx == num_blocks - 1).then_some(record.digest_writes);
                let diff = StateDiff {
                    pre_hi,
                    post_hi,
                    register_reads,
                    digest_writes,
                };
                instruction_blocks.push((instruction, diff, block));
                instruction.remaining_len -= Val::<SC>::from_canonical_usize(KECCAK_RATE_BYTES);
                instruction.src += Val::<SC>::from_canonical_usize(KECCAK_RATE_BYTES);
                instruction.start_timestamp +=
                    Val::<SC>::from_canonical_usize(KECCAK_REGISTER_READS + KECCAK_ABSORB_READS);
            }
        }

        // We need to transpose state matrices due to a plonky3 issue: https://github.com/Plonky3/Plonky3/issues/672
        // Note: the fix for this issue will be a commit after the major Field crate refactor PR https://github.com/Plonky3/Plonky3/pull/640
        //       which will require a significant refactor to switch to.
        let p3_states = states
            .into_iter()
            .map(|state| {
                // transpose of 5x5 matrix
                from_fn(|i| {
                    let x = i / 5;
                    let y = i % 5;
                    state[x + 5 * y]
                })
            })
            .collect();
        let p3_keccak_trace: RowMajorMatrix<Val<SC>> = generate_trace_rows(p3_states, 0);
        let num_rows = p3_keccak_trace.height();
        // Every `NUM_ROUNDS` rows corresponds to one input block
        let num_blocks = num_rows.div_ceil(NUM_ROUNDS);
        // Resize with dummy `is_enabled = 0`
        instruction_blocks.resize(num_blocks, Default::default());

        let aux_cols_factory = memory.aux_cols_factory();

        // Use unsafe alignment so we can parallelly write to the matrix
        let mut trace =
            RowMajorMatrix::new(Val::<SC>::zero_vec(num_rows * trace_width), trace_width);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.air.ptr_max_bits;

        trace
            .values
            .par_chunks_mut(trace_width * NUM_ROUNDS)
            .zip(
                p3_keccak_trace
                    .values
                    .par_chunks(NUM_KECCAK_PERM_COLS * NUM_ROUNDS),
            )
            .zip(instruction_blocks.into_par_iter())
            .for_each(|((rows, p3_keccak_mat), (instruction, diff, block))| {
                let height = rows.len() / trace_width;
                for (row, p3_keccak_row) in rows
                    .chunks_exact_mut(trace_width)
                    .zip(p3_keccak_mat.chunks_exact(NUM_KECCAK_PERM_COLS))
                {
                    // Safety: `KeccakPermCols` **must** be the first field in `KeccakVmCols`
                    row[..NUM_KECCAK_PERM_COLS].copy_from_slice(p3_keccak_row);
                    let row_mut: &mut KeccakVmCols<Val<SC>> = row.borrow_mut();
                    row_mut.instruction = instruction;

                    row_mut.sponge.block_bytes =
                        block.padded_bytes.map(Val::<SC>::from_canonical_u8);
                    if let Some(partial_read_idx) = block.partial_read_idx {
                        let partial_read = memory.record_by_id(block.reads[partial_read_idx]);
                        row_mut
                            .mem_oc
                            .partial_block
                            .copy_from_slice(&partial_read.data_slice()[1..]);
                    }
                    for (i, is_padding) in row_mut.sponge.is_padding_byte.iter_mut().enumerate() {
                        *is_padding = Val::<SC>::from_bool(i >= block.remaining_len);
                    }
                }
                let first_row: &mut KeccakVmCols<Val<SC>> = rows[..trace_width].borrow_mut();
                first_row.sponge.is_new_start = Val::<SC>::from_bool(block.is_new_start);
                first_row.sponge.state_hi = diff.pre_hi.map(Val::<SC>::from_canonical_u8);
                first_row.instruction.is_enabled_first_round = first_row.instruction.is_enabled;
                // Make memory access aux columns. Any aux column not explicitly defined defaults to
                // all 0s
                if let Some(register_reads) = diff.register_reads {
                    let need_range_check = [
                        &register_reads[0], // dst
                        &register_reads[1], // src
                        &register_reads[2], // len
                        &register_reads[2],
                    ]
                    .map(|r| {
                        memory
                            .record_by_id(*r)
                            .data_slice()
                            .last()
                            .unwrap()
                            .as_canonical_u32()
                    });
                    for bytes in need_range_check.chunks(2) {
                        self.bitwise_lookup_chip.request_range(
                            bytes[0] << limb_shift_bits,
                            bytes[1] << limb_shift_bits,
                        );
                    }
                    for (i, id) in register_reads.into_iter().enumerate() {
                        aux_cols_factory.generate_read_aux(
                            memory.record_by_id(id),
                            &mut first_row.mem_oc.register_aux[i],
                        );
                    }
                }
                for (i, id) in block.reads.into_iter().enumerate() {
                    aux_cols_factory.generate_read_aux(
                        memory.record_by_id(id),
                        &mut first_row.mem_oc.absorb_reads[i],
                    );
                }

                let last_row: &mut KeccakVmCols<Val<SC>> =
                    rows[(height - 1) * trace_width..].borrow_mut();
                last_row.sponge.state_hi = diff.post_hi.map(Val::<SC>::from_canonical_u8);
                last_row.inner.export = instruction.is_enabled
                    * Val::<SC>::from_bool(block.remaining_len < KECCAK_RATE_BYTES);
                if let Some(digest_writes) = diff.digest_writes {
                    for (i, record_id) in digest_writes.into_iter().enumerate() {
                        let record = memory.record_by_id(record_id);
                        aux_cols_factory
                            .generate_write_aux(record, &mut last_row.mem_oc.digest_writes[i]);
                    }
                }
            });

        AirProofInput::simple_no_pis(trace)
    }
}

impl<F: PrimeField32> ChipUsageGetter for KeccakVmChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        let num_blocks: usize = self.records.iter().map(|r| r.input_blocks.len()).sum();
        num_blocks * NUM_ROUNDS
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
