use std::{array, borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{ShiftAuxCols, ShiftCols, ShiftIoCols},
    ShiftChip, ShiftRecord,
};
use crate::{
    arch::instructions::{U256Opcode, UsizeOpcode},
    old::uint_multiplication::MemoryData,
};

impl<SC: StarkGenericConfig, const NUM_LIMBS: usize, const LIMB_BITS: usize> Chip<SC>
    for ShiftChip<Val<SC>, NUM_LIMBS, LIMB_BITS>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let aux_cols_factory = self.memory_controller.borrow().aux_cols_factory();

        let width = self.trace_width();
        let height = self.data.len();
        let padded_height = height.next_power_of_two();
        let mut rows = vec![Val::<SC>::zero(); width * padded_height];

        for (row, operation) in rows.chunks_mut(width).zip(self.data) {
            let ShiftRecord::<Val<SC>, NUM_LIMBS, LIMB_BITS> {
                from_state,
                instruction,
                x_ptr_read,
                y_ptr_read,
                z_ptr_read,
                x_read,
                y_read,
                z_write,
                bit_shift_carry,
                bit_shift,
                limb_shift,
                x_sign,
            } = operation;

            let row: &mut ShiftCols<Val<SC>, NUM_LIMBS, LIMB_BITS> = row.borrow_mut();

            row.io = ShiftIoCols {
                from_state: from_state.map(Val::<SC>::from_canonical_u32),
                x: MemoryData::<Val<SC>, NUM_LIMBS, LIMB_BITS> {
                    data: x_read.data,
                    address: x_read.pointer,
                    ptr_to_address: x_ptr_read.pointer,
                },
                y: MemoryData::<Val<SC>, NUM_LIMBS, LIMB_BITS> {
                    data: y_read.data,
                    address: y_read.pointer,
                    ptr_to_address: y_ptr_read.pointer,
                },
                z: MemoryData::<Val<SC>, NUM_LIMBS, LIMB_BITS> {
                    data: z_write.data,
                    address: z_write.pointer,
                    ptr_to_address: z_ptr_read.pointer,
                },
                ptr_as: instruction.d,
                address_as: instruction.e,
            };

            row.aux = ShiftAuxCols {
                is_valid: Val::<SC>::one(),
                bit_shift: Val::<SC>::from_canonical_usize(bit_shift),
                bit_multiplier_left: Val::<SC>::from_canonical_usize(match U256Opcode::from_usize(
                    instruction.opcode,
                ) {
                    U256Opcode::SLL => 1 << bit_shift,
                    U256Opcode::SRL | U256Opcode::SRA => 0,
                    _ => unreachable!(),
                }),
                bit_multiplier_right: Val::<SC>::from_canonical_usize(
                    match U256Opcode::from_usize(instruction.opcode) {
                        U256Opcode::SLL => 0,
                        U256Opcode::SRL | U256Opcode::SRA => 1 << bit_shift,
                        _ => unreachable!(),
                    },
                ),
                x_sign,
                bit_shift_marker: array::from_fn(|val| Val::<SC>::from_bool(val == bit_shift)),
                limb_shift_marker: array::from_fn(|val| Val::<SC>::from_bool(val == limb_shift)),
                bit_shift_carry,
                opcode_sll_flag: Val::<SC>::from_bool(
                    instruction.opcode == U256Opcode::SLL as usize,
                ),
                opcode_srl_flag: Val::<SC>::from_bool(
                    instruction.opcode == U256Opcode::SRL as usize,
                ),
                opcode_sra_flag: Val::<SC>::from_bool(
                    instruction.opcode == U256Opcode::SRA as usize,
                ),
                read_ptr_aux_cols: [z_ptr_read, x_ptr_read, y_ptr_read]
                    .map(|read| aux_cols_factory.make_read_aux_cols(read)),
                read_x_aux_cols: aux_cols_factory.make_read_aux_cols(x_read),
                read_y_aux_cols: aux_cols_factory.make_read_aux_cols(y_read),
                write_z_aux_cols: aux_cols_factory.make_write_aux_cols(z_write),
            };
        }
        AirProofInput::simple_no_pis(air, RowMajorMatrix::new(rows, width))
    }
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> ChipUsageGetter
    for ShiftChip<F, NUM_LIMBS, LIMB_BITS>
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        ShiftCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
