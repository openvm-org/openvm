use openvm_circuit::{
    arch::{Postflight, PostflightError},
    system::memory::{offline_checker::pack_u8_block_bytes, MemoryAuxColsFactory},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::U16_BITS;
use openvm_instructions::LocalOpcode;
use openvm_riscv_circuit::adapters::{ptr_bound_from_ptr, ptr_to_u16_limbs};
use openvm_sha2_air::{set_arrayview_from_u16_le_bytes, set_arrayview_from_u16_slice};
use openvm_stark_backend::{
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
};

use crate::{Sha2ColsRefMut, Sha2Config, Sha2MainChip, Sha2ReplayRow};

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
    let mem_helper = chip.mem_helper.as_borrowed();
    let replay_rows = steps
        .iter()
        .map(|&step| {
            crate::replay_sha2_from_postflight::<F, C>(postflight, step, chip.pointer_max_bits)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (row_index, replay) in replay_rows.iter().enumerate() {
        let row =
            &mut trace.values[row_index * C::MAIN_CHIP_WIDTH..(row_index + 1) * C::MAIN_CHIP_WIDTH];
        chip.fill_trace_row_from_replay(&mem_helper, row, row_index, replay);
    }
    Ok(trace)
}

impl<F: PrimeField32, C: Sha2Config> Sha2MainChip<F, C> {
    fn fill_trace_row_from_replay(
        &self,
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

        for ptr in [replay.dst_ptr, replay.state_ptr, replay.input_ptr] {
            self.range_checker_chip
                .add_count(ptr_bound_from_ptr(ptr, self.pointer_max_bits), U16_BITS);
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
