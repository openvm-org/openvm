use std::{
    array,
    borrow::{Borrow, BorrowMut},
    iter,
};

use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper, U16_BITS,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_cpu_backend::CpuBackend;
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
    system::memory::{
        controller::{dimensions::MemoryDimensions, DIGEST_WIDTH, DIGEST_WIDTH_BITS},
        offline_checker::MemoryBus,
        MemoryAddress, MemoryImage, TimestampedEquipartition,
    },
};

/// Number of memory-bus blocks covered by one merkle leaf.
pub const BLOCKS_PER_LEAF: usize = DIGEST_WIDTH / BLOCK_FE_WIDTH;

/// Number of low bits of a leaf label kept in the `low` limb of [`PersistentBoundaryCols::
/// leaf_label_limbs`].
///
/// A merkle leaf spans `DIGEST_WIDTH` AS-native cells, so the leaf's base cell pointer is
/// `leaf_label * DIGEST_WIDTH`. To send that pointer's low 16-bit limb without composing the full
/// (up to 31-bit) pointer into a field element, we keep the low `LOW_LEAF_BITS` bits of the leaf
/// label in `low` so that `low * DIGEST_WIDTH` (plus a small in-leaf block offset) stays within
/// 16 bits: `LOW_LEAF_BITS = U16_BITS - DIGEST_WIDTH_BITS`.
pub const LOW_LEAF_BITS: usize = U16_BITS - DIGEST_WIDTH_BITS;

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
    /// Leaf label decomposed into little-endian limbs `[low, high]`:
    ///   `leaf_label = low + 2^LOW_LEAF_BITS * high`,
    /// where `low` is range-checked to [`LOW_LEAF_BITS`] bits and `high` to
    /// `address_height - LOW_LEAF_BITS` bits. The decomposition lets us emit the leaf's base
    /// AS-native cell pointer as two 16-bit limbs without composing the full pointer into one
    /// field element.
    pub leaf_label_limbs: [T; 2],
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
    pub range_bus: VariableRangeCheckerBus,
    pub memory_dimensions: MemoryDimensions,
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

        // Decompose the leaf label into `[low, high]` limbs and reconstruct it for the merkle bus.
        // `leaf_label = low + 2^LOW_LEAF_BITS * high`. We range-check the limbs (on active rows
        // only) so that the leaf's base cell pointer `leaf_label * DIGEST_WIDTH` splits cleanly
        // into two 16-bit limbs below: `low * DIGEST_WIDTH < 2^16` and `high < 2^16`.
        let low = local.leaf_label_limbs[0];
        let high = local.leaf_label_limbs[1];
        let leaf_label = low.into() + high.into() * AB::F::from_u32(1u32 << LOW_LEAF_BITS);

        // Active rows have `expand_direction in {1, -1}`, so `expand_direction^2 = 1`; padding rows
        // have `expand_direction = 0`.
        let is_active = local.expand_direction * local.expand_direction;
        let high_bits = self
            .memory_dimensions
            .address_height
            .saturating_sub(LOW_LEAF_BITS);
        self.range_bus
            .range_check(low, LOW_LEAF_BITS)
            .eval(builder, is_active.clone());
        self.range_bus
            .range_check(high, high_bits)
            .eval(builder, is_active);

        let mut expand_fields = vec![
            // direction =  1 => is_final = 0
            // direction = -1 => is_final = 1
            local.expand_direction.into(),
            AB::Expr::ZERO,
            local.address_space - AB::F::from_u32(ADDR_SPACE_OFFSET),
            leaf_label,
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

        for block_idx in 0..BLOCKS_PER_LEAF {
            // The leaf's base cell pointer is `leaf_label * DIGEST_WIDTH`; block `block_idx` starts
            // at `+ block_idx * BLOCK_FE_WIDTH`. As little-endian 16-bit limbs:
            //   pointer_lo = low * DIGEST_WIDTH + block_idx * BLOCK_FE_WIDTH   (< 2^16)
            //   pointer_hi = high
            let pointer_lo = low.into() * AB::F::from_usize(DIGEST_WIDTH)
                + AB::F::from_usize(block_idx * BLOCK_FE_WIDTH);
            let pointer_hi = high.into();
            // Each block uses its own timestamp; untouched blocks stay at t=0.
            self.memory_bus
                .send(
                    MemoryAddress::new(local.address_space.into(), [pointer_lo, pointer_hi]),
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
    range_checker: SharedVariableRangeCheckerChip,
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
    final_memory: &TimestampedEquipartition<F, BLOCK_FE_WIDTH>,
) -> LeafGroupedTouchedMemory<F> {
    let mut enriched: Vec<EnrichedEntry<F>> = final_memory
        .par_iter()
        .map(|&((addr_space, ptr), ts_values)| {
            let leaf_label = ptr / DIGEST_WIDTH as u32;
            let block_idx = ((ptr % DIGEST_WIDTH as u32) / BLOCK_FE_WIDTH as u32) as usize;
            let key = (addr_space, leaf_label);
            let block_info = (block_idx, ts_values.timestamp, ts_values.values);
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
        range_checker: SharedVariableRangeCheckerChip,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        Self {
            air: PersistentBoundaryAir {
                memory_bus,
                merkle_bus,
                compression_bus,
                range_bus: range_checker.bus(),
                memory_dimensions,
            },
            range_checker,
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

            // `leaf_label = low + 2^LOW_LEAF_BITS * high`.
            let low_mask = (1u32 << LOW_LEAF_BITS) - 1;
            let high_bits = self
                .air
                .memory_dimensions
                .address_height
                .saturating_sub(LOW_LEAF_BITS);

            rows.par_chunks_mut(2 * width)
                .zip(touched_labels.par_iter())
                .for_each(|(row, touched_label)| {
                    let low = touched_label.label & low_mask;
                    let high = touched_label.label >> LOW_LEAF_BITS;
                    let leaf_label_limbs = [Val::<SC>::from_u32(low), Val::<SC>::from_u32(high)];
                    // Both the initial and final active rows range-check the limbs (the AIR sends
                    // the range check with multiplicity `expand_direction^2 = 1` on each).
                    for _ in 0..2 {
                        self.range_checker.add_count(low, LOW_LEAF_BITS);
                        self.range_checker.add_count(high, high_bits);
                    }

                    let (initial_row, final_row) = row.split_at_mut(width);
                    *initial_row.borrow_mut() = PersistentBoundaryCols {
                        expand_direction: Val::<SC>::ONE,
                        address_space: Val::<SC>::from_u32(touched_label.address_space),
                        leaf_label_limbs,
                        values: touched_label.init_values,
                        hash: touched_label.init_hash,
                        timestamps: [Val::<SC>::from_u32(INITIAL_TIMESTAMP); BLOCKS_PER_LEAF],
                    };

                    *final_row.borrow_mut() = PersistentBoundaryCols {
                        expand_direction: Val::<SC>::NEG_ONE,
                        address_space: Val::<SC>::from_u32(touched_label.address_space),
                        leaf_label_limbs,
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
