use std::{
    borrow::{Borrow, BorrowMut},
    iter,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::utils::next_power_of_two_or_zero;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rustc_hash::FxHashSet;

use crate::{
    arch::{hasher::HasherChip, POSEIDON2_DIRECT_BUS},
    system::memory::{
        dimensions::MemoryDimensions, manager::memory::INITIAL_TIMESTAMP, merkle::MemoryMerkleBus,
        offline_checker::MemoryBus, Equipartition, MemoryAddress, TimestampedEquipartition,
    },
};

/// The values describe aligned chunk of memory of size `CHUNK`---the data together with the last
/// accessed timestamp---in either the initial or final memory state.
#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct PersistentBoundaryCols<T, const CHUNK: usize> {
    // `expand_direction` =  1 corresponds to initial memory state
    // `expand_direction` = -1 corresponds to final memory state
    // `expand_direction` =  0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub expand_direction: T,
    pub address_space: T,
    pub leaf_label: T,
    pub values: [T; CHUNK],
    pub hash: [T; CHUNK],
    pub timestamp: T,
}

/// Imposes the following constraints:
/// - `expand_direction` should be -1, 0, 1
///
/// Sends the following interactions:
/// - if `expand_direction` is 1, sends `[0, 0, address_space_label, leaf_label]` to `merkle_bus`.
/// - if `expand_direction` is -1, receives `[1, 0, address_space_label, leaf_label]` from `merkle_bus`.
#[derive(Clone, Debug)]
pub struct PersistentBoundaryAir<const CHUNK: usize> {
    pub memory_dims: MemoryDimensions,
    pub memory_bus: MemoryBus,
    pub merkle_bus: MemoryMerkleBus,
}

impl<const CHUNK: usize, F> BaseAir<F> for PersistentBoundaryAir<CHUNK> {
    fn width(&self) -> usize {
        PersistentBoundaryCols::<F, CHUNK>::width()
    }
}

impl<const CHUNK: usize, F> BaseAirWithPublicValues<F> for PersistentBoundaryAir<CHUNK> {}
impl<const CHUNK: usize, F> PartitionedBaseAir<F> for PersistentBoundaryAir<CHUNK> {}

impl<const CHUNK: usize, AB: InteractionBuilder> Air<AB> for PersistentBoundaryAir<CHUNK> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &PersistentBoundaryCols<AB::Var, CHUNK> = (*local).borrow();

        // `direction` should be -1, 0, 1
        builder.assert_eq(
            local.expand_direction,
            local.expand_direction * local.expand_direction * local.expand_direction,
        );

        // TODO[zach]: Make bus interface.
        // Interactions.
        let mut expand_fields = vec![
            // direction =  1 => is_final = 0
            // direction = -1 => is_final = 1
            local.expand_direction.into(),
            AB::Expr::ZERO,
            local.address_space - AB::F::from_canonical_usize(self.memory_dims.as_offset),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.hash.map(Into::into));
        builder.push_send(
            self.merkle_bus.0,
            expand_fields,
            local.expand_direction.into(),
        );

        builder.push_send(
            POSEIDON2_DIRECT_BUS,
            iter::empty()
                .chain(local.values.map(Into::into))
                .chain(iter::repeat(AB::Expr::ZERO).take(CHUNK))
                .chain(local.hash.map(Into::into)),
            local.expand_direction * local.expand_direction,
        );

        self.memory_bus
            .send(
                MemoryAddress::new(
                    local.address_space,
                    local.leaf_label * AB::F::from_canonical_usize(CHUNK),
                ),
                local.values.to_vec(),
                local.timestamp,
            )
            .eval(builder, local.expand_direction);
    }
}

#[derive(Debug)]
pub struct PersistentBoundaryChip<F, const CHUNK: usize> {
    pub air: PersistentBoundaryAir<CHUNK>,
    touched_labels: FxHashSet<(F, usize)>,
}

impl<const CHUNK: usize, F: PrimeField32> PersistentBoundaryChip<F, CHUNK> {
    pub fn new(
        memory_dimensions: MemoryDimensions,
        memory_bus: MemoryBus,
        merkle_bus: MemoryMerkleBus,
    ) -> Self {
        Self {
            air: PersistentBoundaryAir {
                memory_dims: memory_dimensions,
                memory_bus,
                merkle_bus,
            },
            touched_labels: FxHashSet::default(),
        }
    }

    pub fn touch_address(&mut self, address_space: F, pointer: F) {
        let label = pointer.as_canonical_u32() as usize / CHUNK;
        self.touched_labels.insert((address_space, label));
    }

    pub fn current_height(&self) -> usize {
        2 * self.touched_labels.len()
    }

    pub fn generate_trace(
        &self,
        initial_memory: &Equipartition<F, CHUNK>,
        final_memory: &TimestampedEquipartition<F, CHUNK>,
        hasher: &mut impl HasherChip<CHUNK, F>,
    ) -> RowMajorMatrix<F> {
        let width = PersistentBoundaryCols::<F, CHUNK>::width();
        let height = next_power_of_two_or_zero(2 * self.touched_labels.len());
        let mut rows = vec![F::ZERO; height * width];

        for (row, &(address_space, label)) in
            rows.chunks_mut(2 * width).zip(self.touched_labels.iter())
        {
            let (initial_row, final_row) = row.split_at_mut(width);
            *initial_row.borrow_mut() = match initial_memory.get(&(address_space, label)) {
                Some(values) => {
                    let initial_hash = hasher.hash_and_record(values);
                    PersistentBoundaryCols {
                        expand_direction: F::ONE,
                        address_space,
                        leaf_label: F::from_canonical_usize(label),
                        values: *values,
                        hash: initial_hash,
                        timestamp: F::from_canonical_u32(INITIAL_TIMESTAMP),
                    }
                }
                None => {
                    let initial_hash = hasher.hash_and_record(&[F::ZERO; CHUNK]);
                    PersistentBoundaryCols {
                        expand_direction: F::ONE,
                        address_space,
                        leaf_label: F::from_canonical_usize(label),
                        values: [F::ZERO; CHUNK],
                        hash: initial_hash,
                        timestamp: F::ZERO,
                    }
                }
            };
            let timestamped_values = final_memory.get(&(address_space, label)).unwrap();
            let final_hash = hasher.hash_and_record(&timestamped_values.values);
            *final_row.borrow_mut() = PersistentBoundaryCols {
                expand_direction: F::NEG_ONE,
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: timestamped_values.values,
                hash: final_hash,
                timestamp: F::from_canonical_u32(timestamped_values.timestamp),
            };
        }
        RowMajorMatrix::new(rows, width)
    }
}
