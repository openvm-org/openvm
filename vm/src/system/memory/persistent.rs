use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashSet,
};

use afs_derive::AlignedBorrow;
use afs_primitives::utils::next_power_of_two_or_zero;
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::system::memory::{
    dimensions::MemoryDimensions, expand::MemoryMerkleBus, offline_checker::MemoryBus,
    MemoryAddress, MemoryEquipartition,
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
    pub timestamp: T,
}

/// Imposes the following constraints:
/// - `expand_direction` should be -1, 0, 1
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
            AB::Expr::zero(),
            (local.address_space - AB::F::from_canonical_usize(self.memory_dims.as_offset))
                * AB::F::from_canonical_usize(1 << self.memory_dims.address_height),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.values.map(|x| x.into()));
        builder.push_send(
            self.merkle_bus.0,
            expand_fields,
            local.expand_direction.into(),
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

#[derive(Clone, Debug)]
pub struct PersistentBoundaryChip<F, const CHUNK: usize> {
    pub air: PersistentBoundaryAir<CHUNK>,
    touched_labels: HashSet<(F, usize)>,
    initial_memory: MemoryEquipartition<F, CHUNK>,
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
            touched_labels: HashSet::new(),
            initial_memory: MemoryEquipartition::new(),
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
        final_memory: &MemoryEquipartition<F, CHUNK>,
    ) -> RowMajorMatrix<F> {
        let width = PersistentBoundaryCols::<F, CHUNK>::width();
        let height = next_power_of_two_or_zero(2 * self.touched_labels.len());
        let mut rows = vec![F::zero(); height * width];

        for (row, &(address_space, label)) in
            rows.chunks_mut(2 * width).zip(self.touched_labels.iter())
        {
            let (initial_row, final_row) = row.split_at_mut(width);
            *initial_row.borrow_mut() = match self.initial_memory.get(&(address_space, label)) {
                Some(initial) => PersistentBoundaryCols {
                    expand_direction: F::one(),
                    address_space,
                    leaf_label: F::from_canonical_usize(label),
                    values: initial.values,
                    timestamp: F::from_canonical_u32(initial.timestamp),
                },
                None => PersistentBoundaryCols {
                    expand_direction: F::one(),
                    address_space,
                    leaf_label: F::from_canonical_usize(label),
                    values: [F::zero(); CHUNK],
                    timestamp: F::zero(),
                },
            };
            let timestamped_values = final_memory.get(&(address_space, label)).unwrap();
            *final_row.borrow_mut() = PersistentBoundaryCols {
                expand_direction: F::neg_one(),
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: timestamped_values.values,
                timestamp: F::from_canonical_u32(timestamped_values.timestamp),
            };
        }
        RowMajorMatrix::new(rows, width)
    }
}
