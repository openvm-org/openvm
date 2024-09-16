use std::borrow::BorrowMut;

use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{CastFAuxCols, CastFCols, CastFIoCols},
    CastFChip,
};
use crate::{
    arch::chips::MachineChip,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

impl<F: PrimeField32> MachineChip<F> for CastFChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let rows = self
            .data
            .iter()
            .map(|record| {
                let mut row = [F::zero(); CastFCols::<u8>::width()];
                let cols: &mut CastFCols<F> = row[..].borrow_mut();
                cols.io = CastFIoCols {
                    from_state: record.from_state.map(F::from_canonical_usize),
                    op_a: record.instruction.op_a,
                    op_b: record.instruction.op_b,
                    d: record.instruction.d,
                    e: record.instruction.e,
                    x: record.x_read.data,
                    y: record.y_write.data[0],
                };
                cols.aux = CastFAuxCols {
                    is_valid: F::one(),
                    read_x_aux_cols: aux_cols_factory.make_read_aux_cols(record.x_read.clone()),
                    write_y_aux_cols: aux_cols_factory.make_write_aux_cols(record.y_write.clone()),
                };
                row
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();
        let mut blank_row = [F::zero(); CastFCols::<u8>::width()];

        let blank_cols: &mut CastFCols<F> = blank_row[..].borrow_mut();
        *blank_cols = CastFCols::<F> {
            io: Default::default(),
            aux: CastFAuxCols {
                is_valid: Default::default(),
                read_x_aux_cols: MemoryReadAuxCols::disabled(),
                write_y_aux_cols: MemoryWriteAuxCols::disabled(),
            },
        };
        let width = blank_row.len();

        let mut padded_rows = rows;

        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
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
        CastFCols::<F>::width()
    }
}
