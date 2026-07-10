use std::{
    array,
    borrow::{Borrow, BorrowMut},
    iter,
};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::AirProvingContext,
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, Val,
};
use tracing::instrument;

use super::{merkle::SerialReceiver, online::INITIAL_TIMESTAMP};
use crate::{
    arch::{hasher::Hasher, ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH},
    primitives::Chip,
    system::{
        memory::{offline_checker::MemoryBus, MemoryAddress, MemoryImage},
        TouchedMemory,
    },
};

/// Number of memory-bus blocks covered by one merkle leaf.
pub const BLOCKS_PER_LEAF: usize = VM_DIGEST_WIDTH / BLOCK_FE_WIDTH;

/// The values describe one merkle leaf (`DIGEST_WIDTH` cells)---the data together with the
/// last accessed timestamp---in either the initial or final memory state.
#[repr(C)]
#[derive(Debug, AlignedBorrow, StructReflection)]
pub struct PersistentBoundaryCols<T, const DIGEST_WIDTH: usize> {
    // `expand_direction` =  1 corresponds to initial memory state
    // `expand_direction` = -1 corresponds to final memory state
    // `expand_direction` =  0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub expand_direction: T,
    pub address_space: T,
    pub leaf_label: T,
    pub values: [T; DIGEST_WIDTH],
    pub hash: [T; DIGEST_WIDTH],
    /// Per-block timestamps. Each BLOCK_FE_WIDTH block within the leaf has its own timestamp.
    /// For untouched blocks, timestamp stays at 0 (balances: boundary sends at t=0 init, receives
    /// at t=0 final).
    pub timestamps: [T; BLOCKS_PER_LEAF],
}

/// Imposes the following constraints:
/// - `expand_direction` should be -1, 0, 1
///
/// Sends the following interactions:
/// - if `expand_direction` is 1, sends `[0, 0, address_space_label, leaf_label]` to `merkle_bus`.
/// - if `expand_direction` is -1, receives `[1, 0, address_space_label, leaf_label]` from
///   `merkle_bus`.
#[derive(Clone, Debug, ColumnsAir)]
#[columns_via(PersistentBoundaryCols<u8, DIGEST_WIDTH>)]
pub struct PersistentBoundaryAir<const DIGEST_WIDTH: usize> {
    pub memory_bus: MemoryBus,
    pub merkle_bus: PermutationCheckBus,
    pub compression_bus: PermutationCheckBus,
}

impl<const DIGEST_WIDTH: usize, F> BaseAir<F> for PersistentBoundaryAir<DIGEST_WIDTH> {
    fn width(&self) -> usize {
        PersistentBoundaryCols::<F, DIGEST_WIDTH>::width()
    }
}

impl<const DIGEST_WIDTH: usize, F> BaseAirWithPublicValues<F>
    for PersistentBoundaryAir<DIGEST_WIDTH>
{
}
impl<const DIGEST_WIDTH: usize, F> PartitionedBaseAir<F> for PersistentBoundaryAir<DIGEST_WIDTH> {}

impl<const DIGEST_WIDTH: usize, AB: InteractionBuilder> Air<AB>
    for PersistentBoundaryAir<DIGEST_WIDTH>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let local: &PersistentBoundaryCols<AB::Var, DIGEST_WIDTH> = (*local).borrow();

        // `direction` should be -1, 0, 1
        builder.assert_eq(
            local.expand_direction,
            local.expand_direction * local.expand_direction * local.expand_direction,
        );

        // Constrain that an "initial" row has all timestamp zero.
        // Since `direction` is constrained to be in {-1, 0, 1}, we can select `direction == 1`
        // with the constraint below.
        let mut when_initial =
            builder.when(local.expand_direction * (local.expand_direction + AB::F::ONE));
        for i in 0..BLOCKS_PER_LEAF {
            when_initial.assert_zero(local.timestamps[i]);
        }

        let mut expand_fields = vec![
            // direction =  1 => is_final = 0
            // direction = -1 => is_final = 1
            local.expand_direction.into(),
            AB::Expr::ZERO,
            local.address_space - AB::F::from_u32(ADDR_SPACE_OFFSET),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.hash.map(Into::into));
        self.merkle_bus
            .interact(builder, expand_fields, local.expand_direction.into());

        self.compression_bus.interact(
            builder,
            iter::empty()
                .chain(local.values.map(Into::into))
                .chain(iter::repeat_n(AB::Expr::ZERO, DIGEST_WIDTH))
                .chain(local.hash.map(Into::into)),
            local.expand_direction * local.expand_direction,
        );

        let leaf_ptr = local.leaf_label * AB::F::from_usize(DIGEST_WIDTH);
        for block_idx in 0..BLOCKS_PER_LEAF {
            let ptr = leaf_ptr.clone() + AB::F::from_usize(block_idx * BLOCK_FE_WIDTH);
            // Each block uses its own timestamp; untouched blocks stay at t=0.
            self.memory_bus
                .send(
                    MemoryAddress::new(local.address_space, ptr),
                    local.values[block_idx * BLOCK_FE_WIDTH..(block_idx + 1) * BLOCK_FE_WIDTH]
                        .to_vec(),
                    local.timestamps[block_idx],
                )
                .eval(builder, local.expand_direction);
        }
    }
}

pub struct PersistentBoundaryChip<F, const DIGEST_WIDTH: usize> {
    pub air: PersistentBoundaryAir<DIGEST_WIDTH>,
    touched_labels: Option<Vec<FinalTouchedLabel<F, DIGEST_WIDTH>>>,
    overridden_height: Option<usize>,
}

#[derive(Debug)]
pub struct FinalTouchedLabel<F, const DIGEST_WIDTH: usize> {
    address_space: u32,
    label: u32,
    init_values: [F; DIGEST_WIDTH],
    final_values: [F; DIGEST_WIDTH],
    init_hash: [F; DIGEST_WIDTH],
    final_hash: [F; DIGEST_WIDTH],
    /// Per-block timestamps. Each BLOCK_FE_WIDTH block has its own timestamp.
    final_timestamps: [u32; BLOCKS_PER_LEAF],
}

type BlockInfo<F> = (usize, u32, [F; BLOCK_FE_WIDTH]); // (block_idx, timestamp, values)
type EnrichedEntry<F> = ((u32, u32), BlockInfo<F>); // ((addr_space, leaf_label), block_info)
/// Touched memory grouped into merkle leaves: `(addr_space, leaf_label) -> blocks`.
pub(crate) type LeafGroupedTouchedMemory<F> = Vec<((u32, u32), Vec<BlockInfo<F>>)>;

pub(crate) fn group_touched_memory_by_leaf<F: Copy + Send + Sync>(
    final_memory: &TouchedMemory<F>,
) -> LeafGroupedTouchedMemory<F> {
    let mut enriched: Vec<EnrichedEntry<F>> = final_memory
        .par_iter()
        .map(|block| {
            let leaf_label = block.ptr / VM_DIGEST_WIDTH as u32;
            let block_idx = ((block.ptr % VM_DIGEST_WIDTH as u32) / BLOCK_FE_WIDTH as u32) as usize;
            let key = (block.address_space, leaf_label);
            let block_info = (block_idx, block.timestamp, block.values);
            (key, block_info)
        })
        .collect();
    enriched.sort_unstable_by_key(|(key, _)| *key);

    enriched
        .chunk_by(|a, b| a.0 == b.0)
        .map(|group| {
            let key = group[0].0;
            let blocks = group.iter().map(|&(_, info)| info).collect();
            (key, blocks)
        })
        .collect()
}

impl<const DIGEST_WIDTH: usize, F: PrimeField32> PersistentBoundaryChip<F, DIGEST_WIDTH> {
    pub fn new(
        memory_bus: MemoryBus,
        merkle_bus: PermutationCheckBus,
        compression_bus: PermutationCheckBus,
    ) -> Self {
        Self {
            air: PersistentBoundaryAir {
                memory_bus,
                merkle_bus,
                compression_bus,
            },
            touched_labels: None,
            overridden_height: None,
        }
    }

    pub fn set_overridden_height(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }

    /// Finalize the boundary chip with touched memory grouped by Merkle leaf.
    ///
    /// Untouched blocks within a touched leaf get values from initial_memory and timestamp 0.
    #[instrument(name = "boundary_finalize", level = "debug", skip_all)]
    pub(crate) fn finalize<H>(
        &mut self,
        initial_memory: &MemoryImage,
        final_memory_by_leaf: &LeafGroupedTouchedMemory<F>,
        hasher: &H,
    ) where
        H: Hasher<DIGEST_WIDTH, F> + Sync + for<'a> SerialReceiver<&'a [F]>,
    {
        let final_touched_labels: Vec<_> = final_memory_by_leaf
            .par_iter()
            .map(|((addr_space, leaf_label), blocks)| {
                let ptr = leaf_label * DIGEST_WIDTH as u32;
                // SAFETY: addr_space from `final_memory_by_leaf` are all in bounds
                let init_values: [F; DIGEST_WIDTH] = array::from_fn(|i| unsafe {
                    initial_memory.get_f::<F>(*addr_space, ptr + i as u32)
                });

                let mut final_values = init_values;
                let mut timestamps = [0u32; BLOCKS_PER_LEAF];

                for &(block_idx, ts, values) in blocks {
                    timestamps[block_idx] = ts;
                    for (i, &val) in values.iter().enumerate() {
                        final_values[block_idx * BLOCK_FE_WIDTH + i] = val;
                    }
                }

                let initial_hash = hasher.hash(&init_values);
                let final_hash = hasher.hash(&final_values);
                FinalTouchedLabel {
                    address_space: *addr_space,
                    label: *leaf_label,
                    init_values,
                    final_values,
                    init_hash: initial_hash,
                    final_hash,
                    final_timestamps: timestamps,
                }
            })
            .collect();
        for l in &final_touched_labels {
            hasher.receive(&l.init_values);
            hasher.receive(&l.final_values);
        }
        self.touched_labels = Some(final_touched_labels);
    }
}

impl<const DIGEST_WIDTH: usize, RA, SC> Chip<RA, CpuBackend<SC>>
    for PersistentBoundaryChip<Val<SC>, DIGEST_WIDTH>
where
    SC: StarkProtocolConfig,
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let trace = {
            let touched_labels = self
                .touched_labels
                .as_ref()
                .expect("Cannot generate trace before finalization");
            let width = PersistentBoundaryCols::<Val<SC>, DIGEST_WIDTH>::width();
            // Boundary AIR should always present in order to fix the AIR ID of merkle AIR.
            let mut height = (2 * touched_labels.len()).next_power_of_two();
            if let Some(mut oh) = self.overridden_height {
                oh = oh.next_power_of_two();
                assert!(
                    oh >= height,
                    "Overridden height is less than the required height"
                );
                height = oh;
            }
            let mut rows = Val::<SC>::zero_vec(height * width);

            rows.par_chunks_mut(2 * width)
                .zip(touched_labels.par_iter())
                .for_each(|(row, touched_label)| {
                    let (initial_row, final_row) = row.split_at_mut(width);
                    *initial_row.borrow_mut() = PersistentBoundaryCols {
                        expand_direction: Val::<SC>::ONE,
                        address_space: Val::<SC>::from_u32(touched_label.address_space),
                        leaf_label: Val::<SC>::from_u32(touched_label.label),
                        values: touched_label.init_values,
                        hash: touched_label.init_hash,
                        timestamps: [Val::<SC>::from_u32(INITIAL_TIMESTAMP); BLOCKS_PER_LEAF],
                    };

                    *final_row.borrow_mut() = PersistentBoundaryCols {
                        expand_direction: Val::<SC>::NEG_ONE,
                        address_space: Val::<SC>::from_u32(touched_label.address_space),
                        leaf_label: Val::<SC>::from_u32(touched_label.label),
                        values: touched_label.final_values,
                        hash: touched_label.final_hash,
                        timestamps: touched_label.final_timestamps.map(Val::<SC>::from_u32),
                    };
                });
            RowMajorMatrix::new(rows, width)
        };
        AirProvingContext::simple_no_pis(trace)
    }
}
