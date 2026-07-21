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
use rustc_hash::FxHashSet;
use tracing::instrument;

use super::merkle::SerialReceiver;
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

/// Each row describes one touched merkle leaf (`DIGEST_WIDTH` cells): its data and hash in both
/// the initial and final memory state, together with the per-block final timestamps.
/// Initial timestamps are always zero.
#[repr(C)]
#[derive(Debug, AlignedBorrow, StructReflection)]
pub struct PersistentBoundaryCols<T, const DIGEST_WIDTH: usize> {
    pub is_valid: T,
    pub is_dirty: T,
    pub address_space: T,
    pub leaf_label: T,
    pub initial_values: [T; DIGEST_WIDTH],
    pub final_values: [T; DIGEST_WIDTH],
    pub initial_hash: [T; DIGEST_WIDTH],
    pub final_hash: [T; DIGEST_WIDTH],
    /// Per-block timestamps. Each BLOCK_FE_WIDTH block within the leaf has its own timestamp.
    /// For untouched blocks, timestamp stays at 0 (balances: boundary sends at t=0 init, receives
    /// at t=0 final).
    pub final_timestamps: [T; BLOCKS_PER_LEAF],
}

/// Imposes the following constraints:
/// - `is_valid` and `is_dirty` are boolean, and `is_dirty` implies `is_valid`
/// - on clean rows (`is_valid = 1, is_dirty = 0`), final values and hash equal initial ones
///
/// Sends the following interactions (one row per touched leaf):
/// - merkle bus: initial leaf `[1, 0, as_label, leaf_label, initial_hash]` with multiplicity
///   `is_valid`; final leaf `[-1, 0, as_label, leaf_label, final_hash]` with multiplicity
///   `-is_dirty`.
/// - compression bus: `(initial_values, 0, initial_hash)` with multiplicity `is_valid`;
///   `(final_values, 0, final_hash)` with multiplicity `is_dirty`.
/// - memory bus: opens the block at timestamp 0 with `initial_values` (multiplicity `is_valid`) and
///   closes it at `final_timestamps` with `final_values` (multiplicity `-is_valid`).
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

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_dirty);
        // `is_dirty` may only be set on valid rows
        builder
            .when(AB::Expr::ONE - local.is_valid)
            .assert_zero(local.is_dirty);

        // If the leaf is clean, its final values and hash must match the initial ones.
        // Since both bits are boolean and `is_dirty` implies `is_valid`, the selector below
        // is 1 exactly on clean valid rows.
        let mut when_clean = builder.when(local.is_valid - local.is_dirty);
        for i in 0..DIGEST_WIDTH {
            when_clean.assert_eq(local.initial_values[i], local.final_values[i]);
            when_clean.assert_eq(local.initial_hash[i], local.final_hash[i]);
        }

        // merkle-bus interaction: initial leaf
        let mut expand_fields = vec![
            AB::Expr::ONE,
            AB::Expr::ZERO,
            local.address_space - AB::F::from_u32(ADDR_SPACE_OFFSET),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.initial_hash.map(Into::into));
        self.merkle_bus
            .interact(builder, expand_fields, local.is_valid.into());
        // merkle-bus interaction: final leaf
        let mut expand_fields = vec![
            AB::Expr::NEG_ONE,
            AB::Expr::ZERO,
            local.address_space - AB::F::from_u32(ADDR_SPACE_OFFSET),
            local.leaf_label.into(),
        ];
        expand_fields.extend(local.final_hash.map(Into::into));
        self.merkle_bus
            .interact(builder, expand_fields, AB::Expr::ZERO - local.is_dirty);

        // compression bus interaction: initial leaf
        self.compression_bus.interact(
            builder,
            iter::empty()
                .chain(local.initial_values.map(Into::into))
                .chain(iter::repeat_n(AB::Expr::ZERO, DIGEST_WIDTH))
                .chain(local.initial_hash.map(Into::into)),
            local.is_valid,
        );
        // compression bus interaction: final leaf
        self.compression_bus.interact(
            builder,
            iter::empty()
                .chain(local.final_values.map(Into::into))
                .chain(iter::repeat_n(AB::Expr::ZERO, DIGEST_WIDTH))
                .chain(local.final_hash.map(Into::into)),
            local.is_dirty,
        );

        // memory bus interactions
        let leaf_ptr = local.leaf_label * AB::F::from_usize(DIGEST_WIDTH);
        for block_idx in 0..BLOCKS_PER_LEAF {
            let ptr = leaf_ptr.clone() + AB::F::from_usize(block_idx * BLOCK_FE_WIDTH);
            // Each block uses its own timestamp; untouched blocks stay at t=0.
            // initial block
            self.memory_bus
                .send(
                    MemoryAddress::new(local.address_space, ptr.clone()),
                    local.initial_values
                        [block_idx * BLOCK_FE_WIDTH..(block_idx + 1) * BLOCK_FE_WIDTH]
                        .to_vec(),
                    AB::Expr::ZERO,
                )
                .eval(builder, local.is_valid);
            // final block
            self.memory_bus
                .send(
                    MemoryAddress::new(local.address_space, ptr),
                    local.final_values
                        [block_idx * BLOCK_FE_WIDTH..(block_idx + 1) * BLOCK_FE_WIDTH]
                        .to_vec(),
                    local.final_timestamps[block_idx],
                )
                .eval(builder, AB::Expr::ZERO - local.is_valid);
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
    /// Whether the leaf's final values differ from its initial values. Dirtiness gates
    /// all of the row's final-state interactions: a clean leaf's final hash equals its
    /// initial one and is never recorded with the hasher.
    is_dirty: bool,
}

/// Pointers `(address_space, leaf_ptr)` of the touched leaves whose values changed,
/// keyed like `Equipartition<_, DIGEST_WIDTH>` so entries match the merkle chip's
/// `final_memory` keys. Computed by [`PersistentBoundaryChip::finalize`] and consumed by
/// the merkle chip, which emits final-direction rows only along dirty paths.
pub type DirtyLeaves = FxHashSet<(u32, u32)>;

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
    ///
    /// Returns the [`DirtyLeaves`] this chip committed to, so the merkle chip consumes
    /// the exact same bits instead of re-deriving dirtiness.
    #[instrument(name = "boundary_finalize", level = "debug", skip_all)]
    pub(crate) fn finalize<H>(
        &mut self,
        initial_memory: &MemoryImage,
        final_memory_by_leaf: &LeafGroupedTouchedMemory<F>,
        hasher: &H,
    ) -> DirtyLeaves
    where
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

                // Value inequality is the single definition of dirtiness (equal values
                // give equal hashes); a clean leaf's final hash equals the initial one,
                // so the second hash is skipped entirely.
                let is_dirty = final_values != init_values;
                let initial_hash = hasher.hash(&init_values);
                let final_hash = if is_dirty {
                    hasher.hash(&final_values)
                } else {
                    initial_hash
                };
                FinalTouchedLabel {
                    address_space: *addr_space,
                    label: *leaf_label,
                    init_values,
                    final_values,
                    init_hash: initial_hash,
                    final_hash,
                    final_timestamps: timestamps,
                    is_dirty,
                }
            })
            .collect();
        for l in &final_touched_labels {
            hasher.receive(&l.init_values);
            // A clean leaf's final compression has multiplicity `is_dirty = 0` on the
            // compression bus, so it must not be recorded with the hasher chip.
            if l.is_dirty {
                hasher.receive(&l.final_values);
            }
        }
        let dirty_leaves = final_touched_labels
            .iter()
            .filter(|l| l.is_dirty)
            .map(|l| (l.address_space, l.label * DIGEST_WIDTH as u32))
            .collect();
        self.touched_labels = Some(final_touched_labels);
        dirty_leaves
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
            let mut height = touched_labels.len().next_power_of_two();
            if let Some(mut oh) = self.overridden_height {
                oh = oh.next_power_of_two();
                assert!(
                    oh >= height,
                    "Overridden height is less than the required height"
                );
                height = oh;
            }
            let mut rows = Val::<SC>::zero_vec(height * width);

            rows.par_chunks_mut(width)
                .zip(touched_labels.par_iter())
                .for_each(|(row, touched_label)| {
                    *row.borrow_mut() = PersistentBoundaryCols {
                        is_valid: Val::<SC>::ONE,
                        is_dirty: Val::<SC>::from_bool(touched_label.is_dirty),
                        address_space: Val::<SC>::from_u32(touched_label.address_space),
                        leaf_label: Val::<SC>::from_u32(touched_label.label),
                        initial_values: touched_label.init_values,
                        final_values: touched_label.final_values,
                        initial_hash: touched_label.init_hash,
                        final_hash: touched_label.final_hash,
                        final_timestamps: touched_label.final_timestamps.map(Val::<SC>::from_u32),
                    };
                });
            RowMajorMatrix::new(rows, width)
        };
        AirProvingContext::simple_no_pis(trace)
    }
}
