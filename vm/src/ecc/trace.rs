use std::sync::Arc;

use afs_primitives::{ecc::EcAuxCols as EcPrimitiveAuxCols, sub_chip::LocalTraceInstructions};
use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    rap::{get_air_name, AnyRap},
    Chip,
};
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    EcAddUnequalAuxCols, EcAddUnequalChip, EcAddUnequalCols, EcAddUnequalIoCols,
    EcAddUnequalRecord, EcDoubleAuxCols, EcDoubleChip, EcDoubleCols, EcDoubleIoCols,
    EcDoubleRecord, LIMB_SIZE, NUM_LIMBS, TWO_NUM_LIMBS,
};
use crate::{
    arch::{instructions::EccOpcode, MachineChip},
    memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryHeapDataIoCols, MemoryHeapReadRecord,
    },
    utils::limbs_to_biguint,
};

fn load_ec_point<F: PrimeField32>(
    array_read: &MemoryHeapReadRecord<F, TWO_NUM_LIMBS>,
) -> (BigUint, BigUint) {
    let x_limbs = array_read.data_read.data[..NUM_LIMBS]
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect::<Vec<_>>();
    let x = limbs_to_biguint(&x_limbs, LIMB_SIZE);
    let y_limbs = array_read.data_read.data[NUM_LIMBS..]
        .iter()
        .map(|x| x.as_canonical_u32())
        .collect::<Vec<_>>();
    let y = limbs_to_biguint(&y_limbs, LIMB_SIZE);
    (x, y)
}

impl<F: PrimeField32> MachineChip<F> for EcAddUnequalChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        EcAddUnequalCols::<F>::width(&self.air.air.config)
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.config.memory_chip.borrow().aux_cols_factory();

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
                assert_eq!(instruction.opcode, EccOpcode::EC_ADD_NE as usize);

                let io = EcAddUnequalIoCols {
                    from_state: from_state.map(F::from_canonical_usize),
                    p1: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p1_array_read.clone()),
                    p2: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p2_array_read.clone()),
                    p3: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p3_array_write.clone()),
                };

                let (p1_x, p1_y) = load_ec_point(p1_array_read);
                let (p2_x, p2_y) = load_ec_point(p2_array_read);
                let primitive_row = self.air.air.generate_trace_row((
                    (p1_x, p1_y),
                    (p2_x, p2_y),
                    self.config.range_checker_chip.clone(),
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

impl<F: PrimeField32> MachineChip<F> for EcDoubleChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        EcAddUnequalCols::<F>::width(&self.air.air.config)
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let aux_cols_factory = self.config.memory_chip.borrow().aux_cols_factory();

        let rows = self
            .data
            .iter()
            .map(|record| {
                let EcDoubleRecord {
                    from_state,
                    instruction,
                    p1_array_read,
                    p2_array_write,
                } = record;
                assert_eq!(instruction.opcode, EccOpcode::EC_DOUBLE as usize);

                let io = EcDoubleIoCols {
                    from_state: from_state.map(F::from_canonical_usize),
                    p1: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p1_array_read.clone()),
                    p2: MemoryHeapDataIoCols::<F, TWO_NUM_LIMBS>::from(p2_array_write.clone()),
                };
                let (p1_x, p1_y) = load_ec_point(p1_array_read);

                let primitive_row = self
                    .air
                    .air
                    .generate_trace_row(((p1_x, p1_y), self.config.range_checker_chip.clone()));

                let aux = EcDoubleAuxCols {
                    read_p1_aux_cols: aux_cols_factory
                        .make_heap_read_aux_cols(p1_array_read.clone()),
                    write_p2_aux_cols: aux_cols_factory
                        .make_heap_write_aux_cols(p2_array_write.clone()),
                    aux: EcPrimitiveAuxCols {
                        is_valid: F::one(),
                        lambda: primitive_row.aux.lambda,
                        lambda_check: primitive_row.aux.lambda_check,
                        x3_check: primitive_row.aux.x3_check,
                        y3_check: primitive_row.aux.y3_check,
                    },
                };

                EcDoubleCols { io, aux }.flatten()
            })
            .collect::<Vec<_>>();
        let height = rows.len();
        let mut padded_rows = rows;
        let padded_height = height.next_power_of_two();
        let width = EcDoubleCols::<F>::width(&self.air.air.config);

        const IO_WIDTH: usize = EcDoubleIoCols::<u8>::width();
        let dummy_io = [F::zero(); IO_WIDTH];
        let dummy_aux: EcDoubleAuxCols<_> = EcDoubleAuxCols {
            read_p1_aux_cols: MemoryHeapReadAuxCols::disabled(),
            write_p2_aux_cols: MemoryHeapWriteAuxCols::disabled(),
            aux: EcPrimitiveAuxCols::disabled(self.air.air.config.num_limbs),
        };
        let blank_row = [dummy_io.to_vec(), dummy_aux.flatten()].concat();
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for EcAddUnequalChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for EcDoubleChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }
}
