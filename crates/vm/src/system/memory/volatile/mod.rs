use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit_primitives::{
    is_less_than_array::{
        IsLtArrayAuxCols, IsLtArrayIo, IsLtArraySubAir, IsLtArrayWhenTransitionAir,
    },
    utils::implies,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    SubAir, TraceSubRowGenerator,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use rustc_hash::FxHashSet;

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

pub struct VolatileBoundaryChip<F> {
    pub air: VolatileBoundaryAir,
    touched_addresses: FxHashSet<(u32, u32)>,
    range_checker: SharedVariableRangeCheckerChip,
    overridden_height: Option<usize>,
    final_memory: Option<TimestampedEquipartition<F, 1>>,
}

impl<F: Field> VolatileBoundaryChip<F> {
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        range_checker: SharedVariableRangeCheckerChip,
    ) -> Self {
        let range_bus = range_checker.bus();
        Self {
            air: VolatileBoundaryAir::new(
                memory_bus,
                addr_space_max_bits,
                pointer_max_bits,
                range_bus,
            ),
            touched_addresses: FxHashSet::default(),
            range_checker,
            overridden_height: None,
            final_memory: None,
        }
    }

    pub fn touch_address(&mut self, addr_space: u32, pointer: u32) {
        self.touched_addresses.insert((addr_space, pointer));
    }

    pub fn all_addresses(&self) -> Vec<(u32, u32)> {
        self.touched_addresses.iter().cloned().collect()
    }
}

impl<F: PrimeField32> VolatileBoundaryChip<F> {
    pub fn set_overridden_height(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }
    /// Volatile memory requires the starting and final memory to be in equipartition with block size `1`.
    /// When block size is `1`, then the `label` is the same as the address pointer.
    pub fn finalize(&mut self, final_memory: TimestampedEquipartition<F, 1>) {
        self.final_memory = Some(final_memory);
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for VolatileBoundaryChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        // Volatile memory requires the starting and final memory to be in equipartition with block size `1`.
        // When block size is `1`, then the `label` is the same as the address pointer.
        let width = self.trace_width();
        let air = Arc::new(self.air);
        let final_memory = self
            .final_memory
            .expect("Trace generation should be called after finalize");
        let trace_height = if let Some(height) = self.overridden_height {
            assert!(
                height >= final_memory.len(),
                "Overridden height is less than the required height"
            );
            height
        } else {
            final_memory.len()
        };
        let trace_height = trace_height.next_power_of_two();

        // Collect into Vec to sort from BTreeMap and also so we can look at adjacent entries
        let sorted_final_memory: Vec<_> = final_memory.into_par_iter().collect();
        let memory_len = sorted_final_memory.len();

        let mut rows = Val::<SC>::zero_vec(trace_height * width);
        rows.par_chunks_mut(width)
            .zip(sorted_final_memory.par_iter())
            .enumerate()
            .for_each(|(i, (row, ((addr_space, ptr), timestamped_values)))| {
                // `pointer` is the same as `label` since the equipartition has block size 1
                let [data] = timestamped_values.values;
                let row: &mut VolatileBoundaryCols<_> = row.borrow_mut();
                row.addr_space = Val::<SC>::from_canonical_u32(*addr_space);
                row.pointer = Val::<SC>::from_canonical_u32(*ptr);
                row.initial_data = Val::<SC>::ZERO;
                row.final_data = data;
                row.final_timestamp = Val::<SC>::from_canonical_u32(timestamped_values.timestamp);
                row.is_valid = Val::<SC>::ONE;

                // If next.is_valid == 1:
                if i != memory_len - 1 {
                    let (next_addr_space, next_ptr) = sorted_final_memory[i + 1].0;
                    let mut out = Val::<SC>::ZERO;
                    air.addr_lt_air.0.generate_subrow(
                        (
                            self.range_checker.as_ref(),
                            &[row.addr_space, row.pointer],
                            &[
                                Val::<SC>::from_canonical_u32(next_addr_space),
                                Val::<SC>::from_canonical_u32(next_ptr),
                            ],
                        ),
                        ((&mut row.addr_lt_aux).into(), &mut out),
                    );
                    debug_assert_eq!(out, Val::<SC>::ONE, "Addresses are not sorted");
                }
            });
        // Always do a dummy range check on the last row due to wraparound
        if memory_len > 0 {
            let mut out = Val::<SC>::ZERO;
            let row: &mut VolatileBoundaryCols<_> = rows[width * (trace_height - 1)..].borrow_mut();
            air.addr_lt_air.0.generate_subrow(
                (
                    self.range_checker.as_ref(),
                    &[Val::<SC>::ZERO, Val::<SC>::ZERO],
                    &[Val::<SC>::ZERO, Val::<SC>::ZERO],
                ),
                ((&mut row.addr_lt_aux).into(), &mut out),
            );
        }

        let trace = RowMajorMatrix::new(rows, width);
        AirProofInput::simple_no_pis(air, trace)
    }
}

impl<F: PrimeField32> ChipUsageGetter for VolatileBoundaryChip<F> {
    fn air_name(&self) -> String {
        "Boundary".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.touched_addresses.len()
    }

    fn trace_width(&self) -> usize {
        VolatileBoundaryCols::<F>::width()
    }
}
