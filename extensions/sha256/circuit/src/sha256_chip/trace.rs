use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::system::memory::offline_checker::MemoryWriteAuxCols;
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32im_circuit::adapters::compose;
use openvm_sha_air::{get_flag_pt_array, limbs_into_word, Sha256Config, Sha2Air, Sha512Config};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::get_air_name,
    AirRef, Chip, ChipUsageGetter,
};

use super::{Sha2Variant, Sha2VmChip, ShaChipConfig, ShaVmDigestColsRefMut, ShaVmRoundColsRefMut};
use crate::sha256_chip::PaddingFlags;

impl<SC: StarkGenericConfig, C: ShaChipConfig + 'static> Chip<SC> for Sha2VmChip<Val<SC>, C>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let non_padded_height = self.current_trace_height();
        let height = next_power_of_two_or_zero(non_padded_height);
        let width = self.trace_width();
        let mut values = Val::<SC>::zero_vec(height * width);
        if height == 0 {
            return AirProofInput::simple_no_pis(RowMajorMatrix::new(values, width));
        }
        let records = self.records;
        let offline_memory = self.offline_memory.lock().unwrap();
        let memory_aux_cols_factory = offline_memory.aux_cols_factory();

        let mem_ptr_shift: u32 =
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.air.ptr_max_bits);

        let mut states = Vec::with_capacity(height.div_ceil(C::ROWS_PER_BLOCK));
        let mut global_block_idx = 0;
        for (record_idx, record) in records.iter().enumerate() {
            let dst_read = offline_memory.record_by_id(record.dst_read);
            let src_read = offline_memory.record_by_id(record.src_read);
            let len_read = offline_memory.record_by_id(record.len_read);

            self.bitwise_lookup_chip.request_range(
                dst_read
                    .data_at(RV32_REGISTER_NUM_LIMBS - 1)
                    .as_canonical_u32()
                    * mem_ptr_shift,
                src_read
                    .data_at(RV32_REGISTER_NUM_LIMBS - 1)
                    .as_canonical_u32()
                    * mem_ptr_shift,
            );
            let len = compose(len_read.data_slice().try_into().unwrap());
            let mut state = &None;
            for (i, input_message) in record.input_message.iter().enumerate() {
                let input_message = input_message.iter().flatten().copied().collect::<Vec<_>>();
                states.push(Some(Self::generate_state(
                    state,
                    input_message,
                    record_idx,
                    len,
                    i == record.input_records.len() - 1,
                )));
                state = &states[global_block_idx];
                global_block_idx += 1;
            }
        }
        states.extend(
            std::iter::repeat(None).take((height - non_padded_height).div_ceil(C::ROWS_PER_BLOCK)),
        );

        // During the first pass we will fill out most of the matrix
        // But there are some cells that can't be generated by the first pass so we will do a second pass over the matrix
        values
            .par_chunks_mut(width * C::ROWS_PER_BLOCK)
            .zip(states.into_par_iter().enumerate())
            .for_each(|(block, (global_block_idx, state))| {
                // Fill in a valid block
                if let Some(state) = state {
                    let mut has_padding_occurred =
                        state.local_block_idx * C::BLOCK_CELLS > state.message_len as usize;
                    let message_left = if has_padding_occurred {
                        0
                    } else {
                        state.message_len as usize - state.local_block_idx * C::BLOCK_CELLS
                    };
                    let is_last_block = state.is_last_block;
                    let buffer: Vec<Vec<Val<SC>>> = (0..C::MESSAGE_ROWS)
                        .map(|j| {
                            (0..C::CELLS_PER_ROW)
                                .map(|k| {
                                    Val::<SC>::from_canonical_u8(
                                        state.block_input_message[j * C::CELLS_PER_ROW + k],
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();

                    let padded_message: Vec<C::Word> = (0..C::BLOCK_WORDS)
                        .map(|j| {
                            limbs_into_word::<C>(
                                &(0..C::WORD_U8S)
                                    .map(|k| {
                                        state.block_padded_message[(j + 1) * C::WORD_U8S - k - 1]
                                            as u32
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>();

                    self.air.sha_subair.generate_block_trace::<Val<SC>>(
                        block,
                        width,
                        C::VM_CONTROL_WIDTH,
                        &padded_message,
                        self.bitwise_lookup_chip.clone(),
                        &state.hash,
                        is_last_block,
                        global_block_idx as u32 + 1,
                        state.local_block_idx as u32,
                        buffer,
                    );

                    let block_reads = records[state.message_idx].input_records
                        [state.local_block_idx]
                        .iter()
                        .map(|record_id| offline_memory.record_by_id(*record_id))
                        .collect::<Vec<_>>();

                    let mut read_ptr = block_reads[0].pointer;
                    let mut cur_timestamp = Val::<SC>::from_canonical_u32(block_reads[0].timestamp);

                    let read_size = Val::<SC>::from_canonical_usize(C::READ_SIZE);
                    for row in 0..C::ROWS_PER_BLOCK {
                        let row_slice = &mut block[row * width..(row + 1) * width];
                        if row < C::ROUND_ROWS {
                            let mut cols = ShaVmRoundColsRefMut::from::<C>(
                                row_slice[..C::VM_ROUND_WIDTH].borrow_mut(),
                            );
                            *cols.control.len = Val::<SC>::from_canonical_u32(state.message_len);
                            *cols.control.read_ptr = read_ptr;
                            *cols.control.cur_timestamp = cur_timestamp;
                            if row < C::MESSAGE_ROWS {
                                read_ptr += read_size;
                                cur_timestamp += Val::<SC>::ONE;
                                memory_aux_cols_factory
                                    .generate_read_aux(block_reads[row], cols.read_aux);

                                if (row + 1) * C::READ_SIZE <= message_left {
                                    cols.control
                                        .pad_flags
                                        .iter_mut()
                                        .zip(
                                            get_flag_pt_array(
                                                &self.air.padding_encoder,
                                                PaddingFlags::NotPadding as usize,
                                            )
                                            .into_iter()
                                            .map(Val::<SC>::from_canonical_u32),
                                        )
                                        .for_each(|(x, y)| *x = y);
                                } else if !has_padding_occurred {
                                    has_padding_occurred = true;
                                    let len = message_left - row * C::READ_SIZE;
                                    cols.control
                                        .pad_flags
                                        .iter_mut()
                                        .zip(
                                            get_flag_pt_array(
                                                &self.air.padding_encoder,
                                                if row == 3 && is_last_block {
                                                    PaddingFlags::FirstPadding0_LastRow
                                                } else {
                                                    PaddingFlags::FirstPadding0
                                                }
                                                    as usize
                                                    + len,
                                            )
                                            .into_iter()
                                            .map(Val::<SC>::from_canonical_u32),
                                        )
                                        .for_each(|(x, y)| *x = y);
                                } else {
                                    cols.control
                                        .pad_flags
                                        .iter_mut()
                                        .zip(
                                            get_flag_pt_array(
                                                &self.air.padding_encoder,
                                                if row == 3 && is_last_block {
                                                    PaddingFlags::EntirePaddingLastRow
                                                } else {
                                                    PaddingFlags::EntirePadding
                                                }
                                                    as usize,
                                            )
                                            .into_iter()
                                            .map(Val::<SC>::from_canonical_u32),
                                        )
                                        .for_each(|(x, y)| *x = y);
                                }
                            } else {
                                cols.control
                                    .pad_flags
                                    .iter_mut()
                                    .zip(
                                        get_flag_pt_array(
                                            &self.air.padding_encoder,
                                            PaddingFlags::NotConsidered as usize,
                                        )
                                        .into_iter()
                                        .map(Val::<SC>::from_canonical_u32),
                                    )
                                    .for_each(|(x, y)| *x = y);
                            }
                            *cols.control.padding_occurred =
                                Val::<SC>::from_bool(has_padding_occurred);
                        } else {
                            if is_last_block {
                                has_padding_occurred = false;
                            }
                            let mut cols = ShaVmDigestColsRefMut::from::<C>(
                                row_slice[..C::VM_DIGEST_WIDTH].borrow_mut(),
                            );
                            *cols.control.len = Val::<SC>::from_canonical_u32(state.message_len);
                            *cols.control.read_ptr = read_ptr;
                            *cols.control.cur_timestamp = cur_timestamp;
                            cols.control
                                .pad_flags
                                .iter_mut()
                                .zip(
                                    get_flag_pt_array(
                                        &self.air.padding_encoder,
                                        PaddingFlags::NotConsidered as usize,
                                    )
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32),
                                )
                                .for_each(|(x, y)| *x = y);
                            if is_last_block {
                                let record = &records[state.message_idx];
                                let dst_read = offline_memory.record_by_id(record.dst_read);
                                let src_read = offline_memory.record_by_id(record.src_read);
                                let len_read = offline_memory.record_by_id(record.len_read);
                                let digest_writes: Vec<_> = record
                                    .digest_writes
                                    .iter()
                                    .map(|id| offline_memory.record_by_id(*id))
                                    .collect();
                                *cols.from_state = record.from_state;
                                *cols.rd_ptr = dst_read.pointer;
                                *cols.rs1_ptr = src_read.pointer;
                                *cols.rs2_ptr = len_read.pointer;
                                cols.dst_ptr
                                    .iter_mut()
                                    .zip(dst_read.data_slice())
                                    .for_each(|(x, y)| *x = *y);
                                cols.src_ptr
                                    .iter_mut()
                                    .zip(src_read.data_slice())
                                    .for_each(|(x, y)| *x = *y);
                                cols.len_data
                                    .iter_mut()
                                    .zip(len_read.data_slice())
                                    .for_each(|(x, y)| *x = *y);
                                memory_aux_cols_factory.generate_read_aux(
                                    dst_read,
                                    cols.register_reads_aux.get_mut(0).unwrap(),
                                );
                                memory_aux_cols_factory
                                    .generate_read_aux(src_read, &mut cols.register_reads_aux[1]);
                                memory_aux_cols_factory
                                    .generate_read_aux(len_read, &mut cols.register_reads_aux[2]);

                                match C::VARIANT {
                                    Sha2Variant::Sha256 => {
                                        debug_assert_eq!(C::NUM_WRITES, 1);
                                        debug_assert_eq!(digest_writes.len(), 1);
                                        debug_assert_eq!(cols.writes_aux_base.len(), 1);
                                        debug_assert_eq!(cols.writes_aux_prev_data.nrows(), 1);
                                        let digest_write = digest_writes[0];
                                        // write to a temporary MemoryWriteAuxCols object and then copy it over to the columns struct.
                                        let mut writes_aux: MemoryWriteAuxCols<
                                            Val<SC>,
                                            { Sha256Config::DIGEST_SIZE },
                                        > = MemoryWriteAuxCols::from_base(
                                            cols.writes_aux_base[0],
                                            cols.writes_aux_prev_data
                                                .row(0)
                                                .to_vec()
                                                .try_into()
                                                .unwrap(),
                                        );
                                        memory_aux_cols_factory
                                            .generate_write_aux(digest_write, &mut writes_aux);
                                        cols.writes_aux_base[0] = writes_aux.get_base();
                                        cols.writes_aux_prev_data
                                            .row_mut(0)
                                            .iter_mut()
                                            .zip(writes_aux.prev_data())
                                            .for_each(|(x, y)| *x = *y);
                                    }
                                    Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                                        debug_assert_eq!(C::NUM_WRITES, 2);
                                        debug_assert_eq!(digest_writes.len(), 2);
                                        debug_assert_eq!(cols.writes_aux_base.len(), 2);
                                        debug_assert_eq!(cols.writes_aux_prev_data.nrows(), 2);
                                        for (i, digest_write) in digest_writes.iter().enumerate() {
                                            let prev_data =
                                                cols.writes_aux_prev_data.row(i).to_vec();
                                            // write to a temporary MemoryWriteAuxCols object and then copy it over to the columns struct
                                            let mut writes_aux: MemoryWriteAuxCols<
                                                Val<SC>,
                                                { Sha512Config::WRITE_SIZE },
                                            > = MemoryWriteAuxCols::from_base(
                                                cols.writes_aux_base[i],
                                                prev_data.try_into().unwrap(),
                                            );
                                            memory_aux_cols_factory
                                                .generate_write_aux(digest_write, &mut writes_aux);
                                            cols.writes_aux_base[i] = writes_aux.get_base();
                                            cols.writes_aux_prev_data
                                                .row_mut(i)
                                                .iter_mut()
                                                .zip(writes_aux.prev_data())
                                                .for_each(|(x, y)| *x = *y);
                                        }
                                    }
                                }
                            }
                            *cols.control.padding_occurred =
                                Val::<SC>::from_bool(has_padding_occurred);
                        }
                    }
                }
                // Fill in the invalid rows
                else {
                    block.par_chunks_mut(width).for_each(|row| {
                        let cols = ShaVmRoundColsRefMut::from::<C>(row.borrow_mut());
                        self.air.sha_subair.generate_default_row(cols.inner);
                    })
                }
            });

        // Do a second pass over the trace to fill in the missing values
        // Note, we need to skip the very first row
        values[width..]
            .par_chunks_mut(width * C::ROWS_PER_BLOCK)
            .take(non_padded_height / C::ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.air
                    .sha_subair
                    .generate_missing_cells(chunk, width, C::VM_CONTROL_WIDTH);
            });

        AirProofInput::simple_no_pis(RowMajorMatrix::new(values, width))
    }
}

impl<F: PrimeField32, C: ShaChipConfig> ChipUsageGetter for Sha2VmChip<F, C> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.iter().fold(0, |acc, record| {
            acc + record.input_records.len() * C::ROWS_PER_BLOCK
        })
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

/// This is the state information that a block will use to generate its trace
#[derive(Debug, Clone)]
struct Sha2State<C: ShaChipConfig> {
    hash: Vec<C::Word>, // length should be C::HASH_WORDS
    local_block_idx: usize,
    message_len: u32,
    block_input_message: Vec<u8>,  // length should be C::BLOCK_CELLS
    block_padded_message: Vec<u8>, // length should be C::BLOCK_CELLS
    message_idx: usize,
    is_last_block: bool,
}

impl<F: PrimeField32, C: ShaChipConfig> Sha2VmChip<F, C> {
    fn generate_state(
        prev_state: &Option<Sha2State<C>>,
        block_input_message: Vec<u8>, // length should be C::BLOCK_CELLS
        message_idx: usize,
        message_len: u32,
        is_last_block: bool,
    ) -> Sha2State<C> {
        debug_assert_eq!(block_input_message.len(), C::BLOCK_CELLS);
        let local_block_idx = if let Some(prev_state) = prev_state {
            prev_state.local_block_idx + 1
        } else {
            0
        };
        let has_padding_occurred = local_block_idx * C::BLOCK_CELLS > message_len as usize;
        let message_left = if has_padding_occurred {
            0
        } else {
            message_len as usize - local_block_idx * C::BLOCK_CELLS
        };

        let padded_message_bytes: Vec<u8> = (0..C::BLOCK_CELLS)
            .map(|j| {
                if j < message_left {
                    block_input_message[j]
                } else if j == message_left && !has_padding_occurred {
                    1 << (RV32_CELL_BITS - 1)
                } else if !is_last_block || j < C::BLOCK_CELLS - 4 {
                    0u8
                } else {
                    let shift_amount = (C::BLOCK_CELLS - j - 1) * RV32_CELL_BITS;
                    ((message_len * RV32_CELL_BITS as u32)
                        .checked_shr(shift_amount as u32)
                        .unwrap_or(0)
                        & ((1 << RV32_CELL_BITS) - 1)) as u8
                }
            })
            .collect();

        if let Some(prev_state) = prev_state {
            Sha2State {
                hash: Sha2Air::<C>::get_block_hash(
                    &prev_state.hash,
                    prev_state.block_padded_message.clone(),
                ),
                local_block_idx,
                message_len,
                block_input_message,
                block_padded_message: padded_message_bytes,
                message_idx,
                is_last_block,
            }
        } else {
            Sha2State {
                hash: C::get_h().to_vec(),
                local_block_idx: 0,
                message_len,
                block_input_message,
                block_padded_message: padded_message_bytes,
                message_idx,
                is_last_block,
            }
        }
    }
}
