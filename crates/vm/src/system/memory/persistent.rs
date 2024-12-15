use std::{
    borrow::{Borrow, BorrowMut},
    iter,
    sync::Arc,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
#[allow(unused_imports)]
use openvm_stark_backend::p3_maybe_rayon::prelude::IndexedParallelIterator;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{AbstractField, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator, ParallelSliceMut},
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use rustc_hash::FxHashSet;

use super::merkle::DirectCompressionBus;
use crate::{
    arch::hasher::HasherChip,
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
    pub compression_bus: DirectCompressionBus,
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
            self.compression_bus.0,
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
    touched_labels: TouchedLabels<F, CHUNK>,
    overridden_height: Option<usize>,
}

#[derive(Debug)]
enum TouchedLabels<F, const CHUNK: usize> {
    Running(FxHashSet<(F, usize)>),
    Final(Vec<FinalTouchedLabel<F, CHUNK>>),
}

#[derive(Debug)]
struct FinalTouchedLabel<F, const CHUNK: usize> {
    address_space: F,
    label: usize,
    init_values: [F; CHUNK],
    final_values: [F; CHUNK],
    init_exists: bool,
    init_hash: [F; CHUNK],
    final_hash: [F; CHUNK],
    final_timestamp: u32,
}

impl<F: PrimeField32, const CHUNK: usize> Default for TouchedLabels<F, CHUNK> {
    fn default() -> Self {
        Self::Running(FxHashSet::default())
    }
}

impl<F: PrimeField32, const CHUNK: usize> TouchedLabels<F, CHUNK> {
    fn touch(&mut self, address_space: F, label: usize) {
        match self {
            TouchedLabels::Running(touched_labels) => {
                touched_labels.insert((address_space, label));
            }
            _ => panic!("Cannot touch after finalization"),
        }
    }
    fn len(&self) -> usize {
        match self {
            TouchedLabels::Running(touched_labels) => touched_labels.len(),
            TouchedLabels::Final(touched_labels) => touched_labels.len(),
        }
    }
}

impl<const CHUNK: usize, F: PrimeField32> PersistentBoundaryChip<F, CHUNK> {
    pub fn new(
        memory_dimensions: MemoryDimensions,
        memory_bus: MemoryBus,
        merkle_bus: MemoryMerkleBus,
        compression_bus: DirectCompressionBus,
    ) -> Self {
        Self {
            air: PersistentBoundaryAir {
                memory_dims: memory_dimensions,
                memory_bus,
                merkle_bus,
                compression_bus,
            },
            touched_labels: Default::default(),
            overridden_height: None,
        }
    }

    pub fn set_overridden_height(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }

    pub fn touch_address(&mut self, address_space: F, pointer: F) {
        let label = pointer.as_canonical_u32() as usize / CHUNK;
        self.touched_labels.touch(address_space, label);
    }

    pub fn finalize(
        &mut self,
        initial_memory: &Equipartition<F, CHUNK>,
        final_memory: &TimestampedEquipartition<F, CHUNK>,
        hasher: &mut impl HasherChip<CHUNK, F>,
    ) {
        match &mut self.touched_labels {
            TouchedLabels::Running(touched_labels) => {
                // TODO: parallelize this.
                let final_touched_labels = touched_labels
                    .iter()
                    .map(|touched_label| {
                        let (init_exists, initial_hash, init_values) =
                            match initial_memory.get(touched_label) {
                                Some(values) => (true, hasher.hash_and_record(values), *values),
                                None => (
                                    true,
                                    hasher.hash_and_record(&[F::ZERO; CHUNK]),
                                    [F::ZERO; CHUNK],
                                ),
                            };
                        let timestamped_values = final_memory.get(touched_label).unwrap();
                        let final_hash = hasher.hash_and_record(&timestamped_values.values);
                        FinalTouchedLabel {
                            address_space: touched_label.0,
                            label: touched_label.1,
                            init_values,
                            final_values: timestamped_values.values,
                            init_exists,
                            init_hash: initial_hash,
                            final_hash,
                            final_timestamp: timestamped_values.timestamp,
                        }
                    })
                    .collect();
                self.touched_labels = TouchedLabels::Final(final_touched_labels);
            }
            _ => panic!("Cannot finalize after finalization"),
        }
    }
}

impl<const CHUNK: usize, SC: StarkGenericConfig> Chip<SC> for PersistentBoundaryChip<Val<SC>, CHUNK>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = Arc::new(self.air);
        let trace = {
            let width = PersistentBoundaryCols::<Val<SC>, CHUNK>::width();
            // Boundary AIR should always present in order to fix the AIR ID of merkle AIR.
            let mut height = (2 * self.touched_labels.len()).next_power_of_two();
            if let Some(mut oh) = self.overridden_height {
                oh = oh.next_power_of_two();
                assert!(
                    oh >= height,
                    "Overridden height is less than the required height"
                );
                height = oh;
            }
            let mut rows = Val::<SC>::zero_vec(height * width);

            let touched_labels = match self.touched_labels {
                TouchedLabels::Final(touched_labels) => touched_labels,
                _ => panic!("Cannot generate trace before finalization"),
            };

            rows.par_chunks_mut(2 * width)
                .zip(touched_labels.into_par_iter())
                .for_each(|(row, touched_label)| {
                    let (initial_row, final_row) = row.split_at_mut(width);
                    *initial_row.borrow_mut() = PersistentBoundaryCols {
                        expand_direction: Val::<SC>::ONE,
                        address_space: touched_label.address_space,
                        leaf_label: Val::<SC>::from_canonical_usize(touched_label.label),
                        values: touched_label.init_values,
                        hash: touched_label.init_hash,
                        timestamp: if touched_label.init_exists {
                            Val::<SC>::from_canonical_u32(INITIAL_TIMESTAMP)
                        } else {
                            Val::<SC>::ZERO
                        },
                    };

                    *final_row.borrow_mut() = PersistentBoundaryCols {
                        expand_direction: Val::<SC>::NEG_ONE,
                        address_space: touched_label.address_space,
                        leaf_label: Val::<SC>::from_canonical_usize(touched_label.label),
                        values: touched_label.final_values,
                        hash: touched_label.final_hash,
                        timestamp: Val::<SC>::from_canonical_u32(touched_label.final_timestamp),
                    };
                });
            RowMajorMatrix::new(rows, width)
        };
        AirProofInput::simple_no_pis(air, trace)
    }
}

impl<const CHUNK: usize, F: PrimeField32> ChipUsageGetter for PersistentBoundaryChip<F, CHUNK> {
    fn air_name(&self) -> String {
        "Boundary".to_string()
    }

    fn current_trace_height(&self) -> usize {
        2 * self.touched_labels.len()
    }

    fn trace_width(&self) -> usize {
        PersistentBoundaryCols::<F, CHUNK>::width()
    }
}
