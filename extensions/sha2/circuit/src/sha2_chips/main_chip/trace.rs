use std::sync::{atomic::Ordering, Arc};

use openvm_circuit::{
    arch::{Postflight, PostflightError},
    system::memory::{
        offline_checker::pack_u8_block_bytes, MemoryAuxColsFactory, SharedMemoryHelper,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_circuit::adapters::{compute_pointer_carry, ptr_to_u16_limbs};
use openvm_sha2_air::{set_arrayview_from_u16_le_bytes, set_arrayview_from_u16_slice};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};

use crate::{replay_sha2_from_postflight, Sha2ColsRefMut, Sha2Config, Sha2MainChip, Sha2ReplayRow};

pub(crate) fn generate_trace_from_postflight<F, C>(
    chip: &Sha2MainChip<F, C>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError>
where
    F: PrimeField32,
    C: Sha2Config,
{
    let steps = postflight.steps(C::OPCODE.global_opcode());
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace =
        RowMajorMatrix::new(F::zero_vec(height * C::MAIN_CHIP_WIDTH), C::MAIN_CHIP_WIDTH);
    let temporary_range_checker =
        Arc::new(VariableRangeCheckerChip::new(chip.range_checker_chip.bus()));
    let temporary_mem_helper = SharedMemoryHelper::new(
        temporary_range_checker.clone(),
        chip.mem_helper.timestamp_max_bits(),
    );
    let mem_helper = temporary_mem_helper.as_borrowed();
    trace.values[..steps.len() * C::MAIN_CHIP_WIDTH]
        .par_chunks_exact_mut(C::MAIN_CHIP_WIDTH)
        .zip(steps.par_iter().copied())
        .enumerate()
        .try_for_each(|(row_index, (row, step))| {
            let replay =
                replay_sha2_from_postflight::<F, C>(postflight, step, chip.pointer_max_bits)?;
            chip.fill_trace_row_from_replay(
                temporary_range_checker.as_ref(),
                &mem_helper,
                row,
                row_index,
                &replay,
            );
            Ok::<(), PostflightError>(())
        })?;
    if chip.range_checker_chip.count.len() != temporary_range_checker.count.len() {
        return Err(PostflightError::new("SHA-2 range-checker shape mismatch"));
    }
    for (destination, source) in chip
        .range_checker_chip
        .count
        .iter()
        .zip(&temporary_range_checker.count)
    {
        destination.fetch_add(source.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    Ok(trace)
}

#[cfg(test)]
pub(crate) fn generate_trace_from_postflights<F, C>(
    chip: &Sha2MainChip<F, C>,
    postflights: &[Postflight<'_, F>],
) -> Result<RowMajorMatrix<F>, PostflightError>
where
    F: PrimeField32,
    C: Sha2Config,
{
    let mut replay_rows = Vec::new();
    for postflight in postflights {
        for &step in postflight.steps(C::OPCODE.global_opcode()) {
            replay_rows.push(replay_sha2_from_postflight::<F, C>(
                postflight,
                step,
                chip.pointer_max_bits,
            )?);
        }
    }
    Ok(chip.generate_trace_from_replays(&replay_rows))
}

#[cfg(test)]
impl<F: PrimeField32, C: Sha2Config> Sha2MainChip<F, C> {
    fn generate_trace_from_replays(&self, replay_rows: &[Sha2ReplayRow]) -> RowMajorMatrix<F> {
        let height = next_power_of_two_or_zero(replay_rows.len());
        let mut trace =
            RowMajorMatrix::new(F::zero_vec(height * C::MAIN_CHIP_WIDTH), C::MAIN_CHIP_WIDTH);
        let mem_helper = self.mem_helper.as_borrowed();
        trace.values[..replay_rows.len() * C::MAIN_CHIP_WIDTH]
            .par_chunks_exact_mut(C::MAIN_CHIP_WIDTH)
            .zip(replay_rows.par_iter())
            .enumerate()
            .for_each(|(row_index, (row, replay))| {
                self.fill_trace_row_from_replay(
                    self.range_checker_chip.as_ref(),
                    &mem_helper,
                    row,
                    row_index,
                    replay,
                );
            });
        trace
    }
}

impl<F: PrimeField32, C: Sha2Config> Sha2MainChip<F, C> {
    fn fill_trace_row_from_replay(
        &self,
        range_checker: &VariableRangeCheckerChip,
        mem_helper: &MemoryAuxColsFactory<F>,
        row_slice: &mut [F],
        row_idx: usize,
        replay: &Sha2ReplayRow,
    ) {
        let mut cols = Sha2ColsRefMut::from::<C>(row_slice);

        *cols.block.request_id = F::from_usize(row_idx);
        set_arrayview_from_u16_le_bytes(&mut cols.block.message_u16s, &replay.message_bytes);
        set_arrayview_from_u16_le_bytes(&mut cols.block.prev_state, &replay.prev_state);
        set_arrayview_from_u16_le_bytes(&mut cols.block.new_state, &replay.new_state);

        *cols.instruction.is_enabled = F::ONE;
        cols.instruction.from_state.timestamp = F::from_u32(replay.timestamp);
        cols.instruction.from_state.pc = F::from_u32(replay.from_pc);
        *cols.instruction.dst_reg_ptr = F::from_u32(replay.dst_reg_ptr);
        *cols.instruction.state_reg_ptr = F::from_u32(replay.state_reg_ptr);
        *cols.instruction.input_reg_ptr = F::from_u32(replay.input_reg_ptr);

        // Pack low 32 bits of each pointer into u16 cells.
        set_arrayview_from_u16_slice(
            &mut cols.instruction.dst_ptr_limbs,
            ptr_to_u16_limbs(replay.dst_ptr),
        );
        set_arrayview_from_u16_slice(
            &mut cols.instruction.state_ptr_limbs,
            ptr_to_u16_limbs(replay.state_ptr),
        );
        set_arrayview_from_u16_slice(
            &mut cols.instruction.input_ptr_limbs,
            ptr_to_u16_limbs(replay.input_ptr),
        );

        // Byte -> cell pointer conversion carries, plus matching `cell_hi` range-check counts,
        // registered on the caller-provided range checker so error paths stay clean. `replay`
        // holds stable copies of the pointer values, separate from the trace row.
        for (byte_ptr, conv_col) in [
            (replay.input_ptr, cols.mem.input_cell_carry),
            (replay.state_ptr, cols.mem.state_cell_carry),
            (replay.dst_ptr, cols.mem.dst_cell_carry),
        ] {
            *conv_col = F::from_u32(compute_pointer_carry(
                range_checker,
                byte_ptr,
                self.pointer_max_bits,
            ));
        }

        // fill in the register reads aux
        let mut timestamp = replay.timestamp;
        for (cols, &previous_timestamp) in cols
            .mem
            .register_aux
            .iter_mut()
            .zip(replay.register_prev_timestamps.iter())
        {
            mem_helper.fill(previous_timestamp, timestamp, cols.as_mut());
            timestamp += 1;
        }

        replay
            .input_prev_timestamps
            .iter()
            .zip(cols.mem.input_reads)
            .for_each(|(&previous_timestamp, read_aux_cols)| {
                mem_helper.fill(previous_timestamp, timestamp, read_aux_cols.as_mut());
                timestamp += 1;
            });

        replay
            .state_prev_timestamps
            .iter()
            .zip(cols.mem.state_reads)
            .for_each(|(&previous_timestamp, state_aux_cols)| {
                mem_helper.fill(previous_timestamp, timestamp, state_aux_cols.as_mut());
                timestamp += 1;
            });

        replay
            .write_prev_timestamps
            .iter()
            .zip(&replay.write_prev_data)
            .zip(cols.mem.write_aux)
            .for_each(|((&previous_timestamp, previous_data), write_aux_cols)| {
                write_aux_cols.set_prev_data(pack_u8_block_bytes(previous_data));
                mem_helper.fill(previous_timestamp, timestamp, write_aux_cols.as_mut());
                timestamp += 1;
            });
    }
}
