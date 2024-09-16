use afs_primitives::{ecc::EcAuxCols as EcPrimitiveAuxCols, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    EcAddUnequalAuxCols, EcAddUnequalChip, EcAddUnequalCols, EcAddUnequalIoCols, EcAddUnequalRecord,
};
use crate::{
    arch::chips::MachineChip,
    memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryDataIoCols, MemoryHeapDataIoCols,
    },
    modular_arithmetic::{limbs_to_biguint, NUM_LIMBS, TWO_NUM_LIMBS},
};

impl<F: PrimeField32> MachineChip<F> for EcAddUnequalChip<F> {
    fn air<SC: p3_uni_stark::StarkGenericConfig>(
        &self,
    ) -> Box<dyn afs_stark_backend::rap::AnyRap<SC>>
    where
        p3_uni_stark::Domain<SC>: p3_commit::PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        EcAddUnequalCols::<F>::width(&self.air.air.config)
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.memory_chip.borrow().aux_cols_factory();

        let rows = self
            .data
            .iter()
            .map(|record| {
                let EcAddUnequalRecord {
                    from_state,
                    instruction,
                    p1_array_read,
                    p2_array_read,
                    p3_array_write,
                } = record;

                let io = EcAddUnequalIoCols {
                    from_state: from_state.map(F::from_canonical_usize),
                    p1: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p1_array_read.clone()),
                    p2: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p2_array_read.clone()),
                    p3: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p3_array_write.clone()),
                };

                let p1_x_limbs = p1_array_read.data_read.data[..NUM_LIMBS]
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let p1_x = limbs_to_biguint(&p1_x_limbs);
                let p1_y_limbs = p1_array_read.data_read.data[NUM_LIMBS..]
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let p1_y = limbs_to_biguint(&p1_y_limbs);
                let p2_x_limbs = p2_array_read.data_read.data[..NUM_LIMBS]
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let p2_x = limbs_to_biguint(&p2_x_limbs);
                let p2_y_limbs = p2_array_read.data_read.data[NUM_LIMBS..]
                    .iter()
                    .map(|x| x.as_canonical_u32())
                    .collect::<Vec<_>>();
                let p2_y = limbs_to_biguint(&p2_y_limbs);

                let primitive_row = self.air.air.generate_trace_row((
                    (p1_x, p1_y),
                    (p2_x, p2_y),
                    self.range_checker_chip.clone(),
                ));

                let aux = EcAddUnequalAuxCols {
                    read_p1_aux_cols: aux_cols_factory
                        .make_heap_read_aux_cols(p1_array_read.clone()),
                    read_p2_aux_cols: aux_cols_factory
                        .make_heap_read_aux_cols(p2_array_read.clone()),
                    write_p3_aux_cols: aux_cols_factory
                        .make_heap_write_aux_cols(p3_array_write.clone()),
                    aux: EcPrimitiveAuxCols {
                        is_valid: F::one(),
                        lambda: primitive_row.aux.lambda,
                        lambda_check: primitive_row.aux.lambda_check,
                        x3_check: primitive_row.aux.x3_check,
                        y3_check: primitive_row.aux.y3_check,
                    },
                };

                EcAddUnequalCols { io, aux }.flatten()
            })
            .collect::<Vec<_>>();
        let height = rows.len();
        let mut padded_rows = rows;
        let padded_height = height.next_power_of_two();
        let width = EcAddUnequalCols::<F>::width(&self.air.air.config);

        const IO_WIDTH: usize = EcAddUnequalIoCols::<u8>::width();
        let dummy_io = [F::zero(); IO_WIDTH];
        let dummy_aux: EcAddUnequalAuxCols<_> = EcAddUnequalAuxCols {
            read_p1_aux_cols: MemoryHeapReadAuxCols::disabled(),
            read_p2_aux_cols: MemoryHeapReadAuxCols::disabled(),
            write_p3_aux_cols: MemoryHeapWriteAuxCols::disabled(),
            aux: EcPrimitiveAuxCols::disabled(self.air.air.config.num_limbs),
        };
        let blank_row = [dummy_io.to_vec(), dummy_aux.flatten()].concat();
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
