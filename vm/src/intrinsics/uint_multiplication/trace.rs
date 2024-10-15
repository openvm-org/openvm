use std::{array, borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    rap::{get_air_name, AnyRap},
    Chip,
};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{
        MemoryData, UintMultiplicationAuxCols, UintMultiplicationCols, UintMultiplicationIoCols,
    },
    UintMultiplicationChip, UintMultiplicationRecord,
};
use crate::arch::VmChip;

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmChip<F>
    for UintMultiplicationChip<F, NUM_LIMBS, LIMB_BITS>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_controller.borrow().aux_cols_factory();

        let width = self.trace_width();
        let height = self.data.len();
        let padded_height = height.next_power_of_two();
        let mut rows = vec![F::zero(); width * padded_height];

        for (row, operation) in rows.chunks_mut(width).zip(self.data) {
            let UintMultiplicationRecord::<F, NUM_LIMBS, LIMB_BITS> {
                from_state,
                instruction,
                x_ptr_read,
                y_ptr_read,
                z_ptr_read,
                x_read,
                y_read,
                z_write,
                carry,
            } = operation;

            let row: &mut UintMultiplicationCols<F, NUM_LIMBS, LIMB_BITS> = row.borrow_mut();

            row.io = UintMultiplicationIoCols {
                from_state: from_state.map(F::from_canonical_u32),
                x: MemoryData::<F, NUM_LIMBS, LIMB_BITS> {
                    data: x_read.data,
                    address: x_read.pointer,
                    ptr_to_address: x_ptr_read.pointer,
                },
                y: MemoryData::<F, NUM_LIMBS, LIMB_BITS> {
                    data: y_read.data,
                    address: y_read.pointer,
                    ptr_to_address: y_ptr_read.pointer,
                },
                z: MemoryData::<F, NUM_LIMBS, LIMB_BITS> {
                    data: z_write.data,
                    address: z_write.pointer,
                    ptr_to_address: z_ptr_read.pointer,
                },
                ptr_as: instruction.d,
                address_as: instruction.e,
            };

            row.aux = UintMultiplicationAuxCols {
                is_valid: F::one(),
                carry: array::from_fn(|i| carry[i]),
                read_ptr_aux_cols: [z_ptr_read, x_ptr_read, y_ptr_read]
                    .map(|read| aux_cols_factory.make_read_aux_cols(read)),
                read_x_aux_cols: aux_cols_factory.make_read_aux_cols(x_read),
                read_y_aux_cols: aux_cols_factory.make_read_aux_cols(y_read),
                write_z_aux_cols: aux_cols_factory.make_write_aux_cols(z_write),
            };
        }
        RowMajorMatrix::new(rows, width)
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        UintMultiplicationCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<SC: StarkGenericConfig, const NUM_LIMBS: usize, const LIMB_BITS: usize> Chip<SC>
    for UintMultiplicationChip<Val<SC>, NUM_LIMBS, LIMB_BITS>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }
}
