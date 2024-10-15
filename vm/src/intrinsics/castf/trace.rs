use std::{borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    rap::{get_air_name, AnyRap},
    Chip,
};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{CastFAuxCols, CastFCols, CastFIoCols},
    CastFChip,
};
use crate::arch::VmChip;

impl<F: PrimeField32> VmChip<F> for CastFChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_controller.borrow().aux_cols_factory();

        let height = self.data.len();
        let padded_height = height.next_power_of_two();
        let blank_row = [F::zero(); CastFCols::<u8>::width()];
        let mut rows = vec![blank_row; padded_height];
        for (i, record) in self.data.iter().enumerate() {
            let row = &mut rows[i];
            let cols: &mut CastFCols<F> = row[..].borrow_mut();
            cols.io = CastFIoCols {
                from_state: record.from_state.map(F::from_canonical_usize),
                op_a: record.instruction.op_a,
                op_b: record.instruction.op_b,
                d: record.instruction.d,
                e: record.instruction.e,
                x: record.x_write.data,
            };
            cols.aux = CastFAuxCols {
                is_valid: F::one(),
                write_x_aux_cols: aux_cols_factory.make_write_aux_cols(record.x_write.clone()),
                read_y_aux_cols: aux_cols_factory.make_read_aux_cols(record.y_read.clone()),
            };
        }
        RowMajorMatrix::new(rows.concat(), CastFCols::<F>::width())
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        CastFCols::<F>::width()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for CastFChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }
}
