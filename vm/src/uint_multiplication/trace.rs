use afs_stark_backend::{config::StarkGenericConfig, rap::AnyRap};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Domain;

use super::{
    columns::{
        MemoryData, UintMultiplicationAuxCols, UintMultiplicationCols, UintMultiplicationIoCols,
    },
    UintMultiplicationChip,
};
use crate::arch::chips::MachineChip;

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, F: PrimeField32> MachineChip<F>
    for UintMultiplicationChip<NUM_LIMBS, LIMB_BITS, F>
{
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let memory_chip = self.memory_chip.borrow();
        let rows = self
            .data
            .iter()
            .map(|operation| {
                {
                    let super::UintMultiplicationRecord::<NUM_LIMBS, LIMB_BITS, F> {
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

                    UintMultiplicationCols {
                        io: UintMultiplicationIoCols {
                            from_state: from_state.map(F::from_canonical_usize),
                            x: MemoryData::<NUM_LIMBS, LIMB_BITS, F> {
                                data: x_read.data.to_vec(),
                                address: x_read.pointer,
                                ptr_to_address: x_ptr_read.pointer,
                            },
                            y: MemoryData::<NUM_LIMBS, LIMB_BITS, F> {
                                data: y_read.data.to_vec(),
                                address: y_read.pointer,
                                ptr_to_address: y_ptr_read.pointer,
                            },
                            z: MemoryData::<NUM_LIMBS, LIMB_BITS, F> {
                                data: z_write.data.to_vec(),
                                address: z_write.pointer,
                                ptr_to_address: z_ptr_read.pointer,
                            },
                            d: instruction.d,
                            e: instruction.e,
                        },
                        aux: UintMultiplicationAuxCols {
                            is_valid: F::one(),
                            carry: carry.clone(),
                            read_ptr_aux_cols: [z_ptr_read, x_ptr_read, y_ptr_read]
                                .map(|read| memory_chip.make_read_aux_cols(read.clone())),
                            read_x_aux_cols: memory_chip.make_read_aux_cols(x_read.clone()),
                            read_y_aux_cols: memory_chip.make_read_aux_cols(y_read.clone()),
                            write_z_aux_cols: memory_chip.make_write_aux_cols(z_write.clone()),
                        },
                    }
                }
                .flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();
        let blank_row = UintMultiplicationCols::<NUM_LIMBS, LIMB_BITS, F>::default().flatten();
        let padded_rows = rows
            .into_iter()
            .chain(std::iter::repeat(blank_row).take(padded_height - height))
            .collect::<Vec<_>>();
        RowMajorMatrix::new(padded_rows.concat(), self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        self.data.len()
    }

    fn trace_width(&self) -> usize {
        UintMultiplicationCols::<NUM_LIMBS, LIMB_BITS, F>::width()
    }
}
