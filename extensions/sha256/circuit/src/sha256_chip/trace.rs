use std::{array, borrow::BorrowMut, cell::RefCell, sync::Arc};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32im_circuit::adapters::compose;
use openvm_sha256_air::{
    get_flag_pt_array, limbs_into_u32, Sha256Air, SHA256_BLOCK_WORDS, SHA256_BUFFER_SIZE, SHA256_H,
    SHA256_HASH_WORDS, SHA256_ROUNDS_PER_ROW, SHA256_ROWS_PER_BLOCK, SHA256_WORD_U8S,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{AbstractField, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
    },
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};

use super::{
    Sha256VmChip, Sha256VmDigestCols, Sha256VmRoundCols, SHA256VM_CONTROL_WIDTH,
    SHA256VM_DIGEST_WIDTH, SHA256VM_ROUND_WIDTH,
};
use crate::{
    sha256_chip::{PaddingFlags, SHA256_READ_SIZE},
    SHA256_BLOCK_CELLS,
};

impl<SC: StarkGenericConfig> Chip<SC> for Sha256VmChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let non_padded_height = self.current_trace_height();
        let height = next_power_of_two_or_zero(non_padded_height);
        let width = self.trace_width();
        let mut values = Val::<SC>::zero_vec(height * width);
        if height == 0 {
            return AirProofInput::simple(air, RowMajorMatrix::new(values, width), vec![]);
        }
        let records = self.records;
        let memory_aux_cols_factory = RefCell::borrow(&self.memory_controller).aux_cols_factory();

        let mem_ptr_shift: u32 =
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.air.ptr_max_bits);

        let mut states = Vec::with_capacity(non_padded_height / SHA256_ROWS_PER_BLOCK);

        let mut global_block_idx = 0;
        for (record_idx, record) in records.iter().enumerate() {
            self.bitwise_lookup_chip.request_range(
                record.dst_read.data[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32()
                    * mem_ptr_shift,
                record.src_read.data[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32()
                    * mem_ptr_shift,
            );
            let len = compose(record.len_read.data);
            let mut state = None;
            for (i, block_reads) in record.input_message.iter().enumerate() {
                let input_message = array::from_fn(|j| {
                    block_reads[j / (SHA256_ROUNDS_PER_ROW * SHA256_WORD_U8S)].data
                        [j % (SHA256_ROUNDS_PER_ROW * SHA256_WORD_U8S)]
                        .as_canonical_u32() as u8
                });

                states.push(Self::generate_state(
                    state,
                    input_message,
                    record_idx,
                    len,
                    i == record.input_message.len() - 1,
                ));
                state = Some(&states[global_block_idx]);
                global_block_idx += 1;
            }
        }

        values
            .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .zip(states.into_par_iter().enumerate())
            .for_each(|(block, (global_block_idx, state))| {
                let mut has_padding_occurred =
                    state.local_block_idx * SHA256_BLOCK_CELLS > state.message_len as usize;
                let message_left = if has_padding_occurred {
                    0
                } else {
                    state.message_len as usize - state.local_block_idx * SHA256_BLOCK_CELLS
                };
                let is_last_block = state.is_last_block;
                let buffer: [[Val<SC>; SHA256_BUFFER_SIZE]; 4] = array::from_fn(|j| {
                    array::from_fn(|k| {
                        Val::<SC>::from_canonical_u8(
                            state.block_input_message[j * SHA256_BUFFER_SIZE + k],
                        )
                    })
                });

                let padded_message: [u32; SHA256_BLOCK_WORDS] = array::from_fn(|j| {
                    limbs_into_u32::<RV32_REGISTER_NUM_LIMBS>(array::from_fn(|k| {
                        state.block_padded_message[(j + 1) * SHA256_WORD_U8S - k - 1] as u32
                    }))
                });

                self.air.sha256_subair.generate_block_trace::<Val<SC>>(
                    block,
                    width,
                    SHA256VM_CONTROL_WIDTH,
                    &padded_message,
                    self.bitwise_lookup_chip.as_ref(),
                    &state.hash,
                    is_last_block,
                    global_block_idx as u32 + 1,
                    state.local_block_idx as u32,
                    &buffer,
                );

                let block_reads = &records[state.message_idx].input_message[state.local_block_idx];

                let mut read_ptr = block_reads[0].pointer;
                let mut cur_timestamp = Val::<SC>::from_canonical_u32(block_reads[0].timestamp);

                let read_size = Val::<SC>::from_canonical_usize(SHA256_READ_SIZE);
                for row in 0..SHA256_ROWS_PER_BLOCK {
                    let row_slice = &mut block[row * width..(row + 1) * width];
                    if row < 16 {
                        let cols: &mut Sha256VmRoundCols<Val<SC>> =
                            row_slice[..SHA256VM_ROUND_WIDTH].borrow_mut();
                        cols.control.len = Val::<SC>::from_canonical_u32(state.message_len);
                        cols.control.read_ptr = read_ptr;
                        cols.control.cur_timestamp = cur_timestamp;
                        if row < 4 {
                            read_ptr += read_size;
                            cur_timestamp += Val::<SC>::ONE;
                            cols.read_aux =
                                memory_aux_cols_factory.make_read_aux_cols(block_reads[row]);

                            if (row + 1) * SHA256_READ_SIZE <= message_left {
                                cols.control.pad_flags = get_flag_pt_array(
                                    &self.air.padding_encoder,
                                    PaddingFlags::NotPadding as usize,
                                )
                                .map(Val::<SC>::from_canonical_u32);
                            } else if !has_padding_occurred {
                                has_padding_occurred = true;
                                let len = message_left - row * SHA256_READ_SIZE;
                                cols.control.pad_flags = get_flag_pt_array(
                                    &self.air.padding_encoder,
                                    if row == 3 && is_last_block {
                                        PaddingFlags::FirstPadding0_LastRow
                                    } else {
                                        PaddingFlags::FirstPadding0
                                    } as usize
                                        + len,
                                )
                                .map(Val::<SC>::from_canonical_u32);
                            } else {
                                cols.control.pad_flags = get_flag_pt_array(
                                    &self.air.padding_encoder,
                                    if row == 3 && is_last_block {
                                        PaddingFlags::EntirePaddingLastRow
                                    } else {
                                        PaddingFlags::EntirePadding
                                    } as usize,
                                )
                                .map(Val::<SC>::from_canonical_u32);
                            }
                        } else {
                            cols.control.pad_flags = get_flag_pt_array(
                                &self.air.padding_encoder,
                                PaddingFlags::NotConsidered as usize,
                            )
                            .map(Val::<SC>::from_canonical_u32);
                        }
                        cols.control.padding_occurred = Val::<SC>::from_bool(has_padding_occurred);
                    } else {
                        if is_last_block {
                            has_padding_occurred = false;
                        }
                        let cols: &mut Sha256VmDigestCols<Val<SC>> =
                            row_slice[..SHA256VM_DIGEST_WIDTH].borrow_mut();
                        cols.control.len = Val::<SC>::from_canonical_u32(state.message_len);
                        cols.control.read_ptr = read_ptr;
                        cols.control.cur_timestamp = cur_timestamp;
                        cols.control.pad_flags = get_flag_pt_array(
                            &self.air.padding_encoder,
                            PaddingFlags::NotConsidered as usize,
                        )
                        .map(Val::<SC>::from_canonical_u32);
                        if is_last_block {
                            let record = &records[state.message_idx];
                            cols.from_state = record.from_state;
                            cols.rd_ptr = record.dst_read.pointer;
                            cols.rs1_ptr = record.src_read.pointer;
                            cols.rs2_ptr = record.len_read.pointer;
                            cols.dst_ptr = record.dst_read.data;
                            cols.src_ptr = record.src_read.data;
                            cols.len_data = record.len_read.data;
                            cols.register_reads_aux = [
                                memory_aux_cols_factory.make_read_aux_cols(record.dst_read),
                                memory_aux_cols_factory.make_read_aux_cols(record.src_read),
                                memory_aux_cols_factory.make_read_aux_cols(record.len_read),
                            ];
                            cols.writes_aux =
                                memory_aux_cols_factory.make_write_aux_cols(record.digest_write);
                        }
                        cols.control.padding_occurred = Val::<SC>::from_bool(has_padding_occurred);
                    }
                }
            });

        // Fill in the padding rows
        for i in non_padded_height..height {
            let rows = &mut values[(i - 1) * width..(i + 1) * width];
            let (local, next) = rows.split_at_mut(width);
            let local_cols: &mut Sha256VmRoundCols<Val<SC>> = local.borrow_mut();
            let next_cols: &mut Sha256VmRoundCols<Val<SC>> = next.borrow_mut();
            self.air
                .sha256_subair
                .default_row(&local_cols.inner, &mut next_cols.inner);
        }

        // Fill in the w_3 and intermed_4
        for i in 0..height - 1 {
            let rows = &mut values[i * width..(i + 2) * width];
            let (local, next) = rows.split_at_mut(width);
            let local_cols: &mut Sha256VmRoundCols<Val<SC>> = local.borrow_mut();
            let next_cols: &mut Sha256VmRoundCols<Val<SC>> = next.borrow_mut();
            Sha256Air::generate_w_3::<Val<SC>>(&local_cols.inner, &mut next_cols.inner);
            Sha256Air::generate_intermed_4::<Val<SC>>(&local_cols.inner, &mut next_cols.inner);
        }
        // Fill in w_3 and intermed_4 for the last row
        let (first, rest) = values.split_at_mut(width);
        let rest_len = rest.len();
        let last = &mut rest[rest_len - width..];
        let local_cols: &mut Sha256VmRoundCols<Val<SC>> = last.borrow_mut();
        let next_cols: &mut Sha256VmRoundCols<Val<SC>> = first.borrow_mut();
        Sha256Air::generate_w_3::<Val<SC>>(&local_cols.inner, &mut next_cols.inner);
        Sha256Air::generate_intermed_4::<Val<SC>>(&local_cols.inner, &mut next_cols.inner);

        // Fill in intermed_8
        for i in 0..height - 1 {
            let rows = &mut values[i * width..(i + 2) * width];
            let (local, next) = rows.split_at_mut(width);
            let local_cols: &mut Sha256VmRoundCols<Val<SC>> = local.borrow_mut();
            let next_cols: &mut Sha256VmRoundCols<Val<SC>> = next.borrow_mut();
            Sha256Air::generate_intermed_8::<Val<SC>>(&local_cols.inner, &mut next_cols.inner);
        }

        // Fill in intermed_8 for the last row
        let (first, rest) = values.split_at_mut(width);
        let rest_len = rest.len();
        let last = &mut rest[rest_len - width..];
        let local_cols: &mut Sha256VmRoundCols<Val<SC>> = last.borrow_mut();
        let next_cols: &mut Sha256VmRoundCols<Val<SC>> = first.borrow_mut();
        Sha256Air::generate_intermed_8::<Val<SC>>(&local_cols.inner, &mut next_cols.inner);

        // Fill in intermed_12
        for i in 0..height - 1 {
            let rows = &mut values[i * width..(i + 2) * width];
            let (local, next) = rows.split_at_mut(width);
            let local_cols: &mut Sha256VmRoundCols<Val<SC>> = local.borrow_mut();
            let next_cols: &mut Sha256VmRoundCols<Val<SC>> = next.borrow_mut();
            Sha256Air::generate_intermed_12::<Val<SC>>(&mut local_cols.inner, &next_cols.inner);
        }

        // Fill in intermed_12 for the last row
        let (first, rest) = values.split_at_mut(width);
        let rest_len = rest.len();
        let last = &mut rest[rest_len - width..];
        let local_cols: &mut Sha256VmRoundCols<Val<SC>> = last.borrow_mut();
        let next_cols: &mut Sha256VmRoundCols<Val<SC>> = first.borrow_mut();
        Sha256Air::generate_intermed_12::<Val<SC>>(&mut local_cols.inner, &next_cols.inner);

        AirProofInput::simple(air, RowMajorMatrix::new(values, width), vec![])
    }
}

impl<F: PrimeField32> ChipUsageGetter for Sha256VmChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.iter().fold(0, |acc, record| {
            acc + record.input_message.len() * SHA256_ROWS_PER_BLOCK
        })
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

/// This is the minimal state information that a block needs to generate its trace
#[derive(Debug, Clone)]
struct Sha256State {
    hash: [u32; SHA256_HASH_WORDS],
    local_block_idx: usize,
    message_len: u32,
    block_input_message: [u8; SHA256_BLOCK_CELLS],
    block_padded_message: [u8; SHA256_BLOCK_CELLS],
    message_idx: usize,
    is_last_block: bool,
}

impl<F: PrimeField32> Sha256VmChip<F> {
    fn generate_state(
        prev_state: Option<&Sha256State>,
        block_input_message: [u8; SHA256_BLOCK_CELLS],
        message_idx: usize,
        message_len: u32,
        is_last_block: bool,
    ) -> Sha256State {
        let local_block_idx = if let Some(prev_state) = prev_state {
            prev_state.local_block_idx + 1
        } else {
            0
        };
        let has_padding_occurred = local_block_idx * SHA256_BLOCK_CELLS > message_len as usize;
        let message_left = if has_padding_occurred {
            0
        } else {
            message_len as usize - local_block_idx * SHA256_BLOCK_CELLS
        };

        let padded_message_bytes: [u8; SHA256_BLOCK_CELLS] = array::from_fn(|j| {
            if j < message_left {
                block_input_message[j]
            } else if j == message_left && !has_padding_occurred {
                1 << (RV32_CELL_BITS - 1)
            } else if !is_last_block || j < SHA256_BLOCK_CELLS - 4 {
                0u8
            } else {
                let shift_amount = (SHA256_BLOCK_CELLS - j - 1) * RV32_CELL_BITS;
                ((message_len * RV32_CELL_BITS as u32)
                    .checked_shr(shift_amount as u32)
                    .unwrap_or(0)
                    & ((1 << RV32_CELL_BITS) - 1)) as u8
            }
        });

        let state = if let Some(prev_state) = prev_state {
            Sha256State {
                hash: Sha256Air::get_block_hash(&prev_state.hash, prev_state.block_padded_message),
                local_block_idx,
                message_len,
                block_input_message,
                block_padded_message: padded_message_bytes,
                message_idx,
                is_last_block,
            }
        } else {
            Sha256State {
                hash: SHA256_H,
                local_block_idx: 0,
                message_len,
                block_input_message,
                block_padded_message: padded_message_bytes,
                message_idx,
                is_last_block,
            }
        };
        state
    }
}
