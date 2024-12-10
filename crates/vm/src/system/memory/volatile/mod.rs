use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashSet,
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::{
    is_less_than_array::{
        IsLtArrayAuxCols, IsLtArrayIo, IsLtArraySubAir, IsLtArrayWhenTransitionAir,
    },
    utils::implies,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};
use ax_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{AbstractField, Field, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::TimestampedEquipartition;
use crate::system::memory::{
    offline_checker::{MemoryBus, AUX_LEN},
    MemoryAddress,
};

#[cfg(test)]
mod tests;

/// Address stored as address space, pointer
const ADDR_ELTS: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct VolatileBoundaryCols<T> {
    pub addr_space: T,
    pub pointer: T,

    pub initial_data: T,
    pub final_data: T,
    pub final_timestamp: T,

    /// Boolean. `1` if a non-padding row with a valid touched address, `0` if it is a padding row.
    pub is_valid: T,
    pub addr_lt_aux: IsLtArrayAuxCols<T, ADDR_ELTS, AUX_LEN>,
}

#[derive(Clone, Debug)]
pub struct VolatileBoundaryAir {
    pub memory_bus: MemoryBus,
    pub addr_lt_air: IsLtArrayWhenTransitionAir<ADDR_ELTS>,
}

impl VolatileBoundaryAir {
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        let addr_lt_air =
            IsLtArraySubAir::<ADDR_ELTS>::new(range_bus, addr_space_max_bits.max(pointer_max_bits))
                .when_transition();
        Self {
            memory_bus,
            addr_lt_air,
        }
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for VolatileBoundaryAir {}
impl<F: Field> PartitionedBaseAir<F> for VolatileBoundaryAir {}
impl<F: Field> BaseAir<F> for VolatileBoundaryAir {
    fn width(&self) -> usize {
        VolatileBoundaryCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for VolatileBoundaryAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let [local, next] = [0, 1].map(|i| main.row_slice(i));
        let local: &VolatileBoundaryCols<_> = (*local).borrow();
        let next: &VolatileBoundaryCols<_> = (*next).borrow();

        builder.assert_bool(local.is_valid);

        // Ensuring all non-padding rows are at the bottom
        builder
            .when_transition()
            .assert_one(implies(next.is_valid, local.is_valid));

        // Assert local addr < next addr when next.is_valid
        // This ensures the addresses in non-padding rows are all sorted
        let lt_io = IsLtArrayIo {
            x: [local.addr_space, local.pointer].map(Into::into),
            y: [next.addr_space, next.pointer].map(Into::into),
            out: AB::Expr::ONE,
            count: next.is_valid.into(),
        };
        // N.B.: this will do range checks (but not other constraints) on the last row if the first row has is_valid = 1 due to wraparound
        self.addr_lt_air
            .eval(builder, (lt_io, (&local.addr_lt_aux).into()));

        // Write the initial memory values at initial timestamps
        self.memory_bus
            .send(
                MemoryAddress::new(local.addr_space, local.pointer),
                vec![local.initial_data],
                AB::Expr::ZERO,
            )
            .eval(builder, local.is_valid);

        // Read the final memory values at last timestamps when written to
        self.memory_bus
            .receive(
                MemoryAddress::new(local.addr_space, local.pointer),
                vec![local.final_data],
                local.final_timestamp,
            )
            .eval(builder, local.is_valid);
    }
}

#[derive(Debug)]
pub struct VolatileBoundaryChip<F> {
    pub air: VolatileBoundaryAir,
    touched_addresses: HashSet<(F, F)>,
    range_checker: Arc<VariableRangeCheckerChip>,
}

impl<F: Field> VolatileBoundaryChip<F> {
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        let range_bus = range_checker.bus();
        Self {
            air: VolatileBoundaryAir::new(
                memory_bus,
                addr_space_max_bits,
                pointer_max_bits,
                range_bus,
            ),
            touched_addresses: HashSet::new(),
            range_checker,
        }
    }

    pub fn touch_address(&mut self, addr_space: F, pointer: F) {
        self.touched_addresses.insert((addr_space, pointer));
    }

    pub fn all_addresses(&self) -> Vec<(F, F)> {
        self.touched_addresses.iter().cloned().collect()
    }

    pub fn current_height(&self) -> usize {
        self.touched_addresses.len()
    }
}

impl<F: PrimeField32> VolatileBoundaryChip<F> {
    /// Volatile memory requires the starting and final memory to be in equipartition with block size `1`.
    /// When block size is `1`, then the `label` is the same as the address pointer.
    pub fn generate_trace(
        &self,
        final_memory: &TimestampedEquipartition<F, 1>,
        overridden_height: Option<usize>,
    ) -> RowMajorMatrix<F> {
        let trace_height = if let Some(height) = overridden_height {
            assert!(
                height >= final_memory.len(),
                "Overridden height is less than the required height"
            );
            height
        } else {
            final_memory.len()
        };
        self.generate_trace_with_height(final_memory, trace_height.next_power_of_two())
    }

    fn generate_trace_with_height(
        &self,
        final_memory: &TimestampedEquipartition<F, 1>,
        trace_height: usize,
    ) -> RowMajorMatrix<F> {
        assert!(trace_height.is_power_of_two());
        let width = BaseAir::<F>::width(&self.air);

        // Collect into Vec to sort from BTreeMap and also so we can look at adjacent entries
        let sorted_final_memory: Vec<_> = final_memory.iter().collect();
        assert!(sorted_final_memory.len() <= trace_height);

        let mut rows = F::zero_vec(trace_height * width);
        rows.par_chunks_mut(width)
            .zip(&sorted_final_memory)
            .enumerate()
            .for_each(|(i, (row, ((addr_space, ptr), timestamped_values)))| {
                // `pointer` is the same as `label` since the equipartition has block size 1
                let [data] = timestamped_values.values;
                let row: &mut VolatileBoundaryCols<_> = row.borrow_mut();
                row.addr_space = *addr_space;
                row.pointer = F::from_canonical_usize(*ptr);
                row.initial_data = F::ZERO;
                row.final_data = data;
                row.final_timestamp = F::from_canonical_u32(timestamped_values.timestamp);
                row.is_valid = F::ONE;

                // If next.is_valid == 1:
                if i != sorted_final_memory.len() - 1 {
                    let (next_addr_space, next_ptr) = *sorted_final_memory[i + 1].0;
                    let mut out = F::ZERO;
                    self.air.addr_lt_air.0.generate_subrow(
                        (
                            &self.range_checker,
                            &[row.addr_space, row.pointer],
                            &[next_addr_space, F::from_canonical_usize(next_ptr)],
                        ),
                        ((&mut row.addr_lt_aux).into(), &mut out),
                    );
                    debug_assert_eq!(out, F::ONE, "Addresses are not sorted");
                }
            });
        // Always do a dummy range check on the last row due to wraparound
        if !sorted_final_memory.is_empty() {
            let mut out = F::ZERO;
            let row: &mut VolatileBoundaryCols<_> = rows[width * (trace_height - 1)..].borrow_mut();
            self.air.addr_lt_air.0.generate_subrow(
                (
                    &self.range_checker,
                    &[F::ZERO, F::ZERO],
                    &[F::ZERO, F::ZERO],
                ),
                ((&mut row.addr_lt_aux).into(), &mut out),
            );
        }

        RowMajorMatrix::new(rows, width)
    }
}
