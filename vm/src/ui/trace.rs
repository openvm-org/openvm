use std::borrow::BorrowMut;

use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{UiAuxCols, UiCols, UiIoCols},
    UiChip,
};
use crate::arch::chips::MachineChip;

impl<F: PrimeField32> MachineChip<F> for UiChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let height = self.data.len();
        let padded_height = height.next_power_of_two();
        let blank_row = [F::zero(); UiCols::<u8>::width()];
        let mut rows = vec![blank_row; padded_height];
        for (i, record) in self.data.iter().enumerate() {
            let row = &mut rows[i];
            let cols: &mut UiCols<F> = row[..].borrow_mut();
            cols.io = UiIoCols {
                from_state: record.from_state.map(F::from_canonical_usize),
                op_a: record.instruction.op_a,
                op_b: record.instruction.op_b,
                x_cols: [record.x_write.data[2], record.x_write.data[3]],
            };
            cols.aux = UiAuxCols {
                is_valid: F::one(),
                imm_lo_hex: F::from_canonical_u32(
                    F::as_canonical_u32(&record.x_write.data[1]) >> 4,
                ),
                write_x_aux_cols: aux_cols_factory.make_write_aux_cols(record.x_write.clone()),
            };
        }
        RowMajorMatrix::new(rows.concat(), UiCols::<F>::width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        UiCols::<F>::width()
    }
}
