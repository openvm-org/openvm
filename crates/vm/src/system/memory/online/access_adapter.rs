use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    is_less_than::{IsLessThanIo, IsLtSubAir},
    var_range::SharedVariableRangeCheckerChip,
    AlignedBorrow, SubAir, TraceSubRowGenerator,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::system::memory::{
    offline_checker::{MemoryBus, AUX_LEN},
    MemoryAddress,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
/// The columns of the AccessAdapterAir, indicating one split or merge operation.
pub(crate) struct AccessAdapterCols<T, const LEN: usize, const NUM_BLOCKS: usize> {
    addr_space: T,
    pointer: T,
    /// The values of the whole block being split or merged.
    values: [T; LEN],
    /// The timestamps of the blocks being split or merged.
    block_timestamps: [T; NUM_BLOCKS],
    /// The timestamp of the current operation.
    timestamp: T,
    /// The direction of the current operation -- 1 means split, -1 means merge, 0 means dummy row.
    dir: T,
    /// The auxiliary columns for the `is_less_than` constraints.
    lt_aux: [[T; AUX_LEN]; NUM_BLOCKS],
}

#[derive(Clone, Debug)]
pub(crate) struct AccessAdapterAir<const LEN: usize, const NUM_BLOCKS: usize> {
    pub memory_bus: MemoryBus,
    pub lt_air: IsLtSubAir,
}

impl<T, const LEN: usize, const NUM_BLOCKS: usize> BaseAirWithPublicValues<T>
    for AccessAdapterAir<LEN, NUM_BLOCKS>
{
    // No public values associated with this AIR directly for now.
}

impl<T, const LEN: usize, const NUM_BLOCKS: usize> PartitionedBaseAir<T>
    for AccessAdapterAir<LEN, NUM_BLOCKS>
{
    // This AIR is not partitioned.
}

impl<T, const LEN: usize, const NUM_BLOCKS: usize> BaseAir<T>
    for AccessAdapterAir<LEN, NUM_BLOCKS>
{
    fn width(&self) -> usize {
        // The width corresponds to the number of columns in AccessAdapterCols.
        // We use u8 as a placeholder type for size calculation.
        core::mem::size_of::<AccessAdapterCols<u8, LEN, NUM_BLOCKS>>()
    }
}

impl<const LEN: usize, const NUM_BLOCKS: usize, AB: InteractionBuilder> Air<AB>
    for AccessAdapterAir<LEN, NUM_BLOCKS>
{
    fn eval(&self, builder: &mut AB) {
        // We want to:
        // 1. Interact over the memory bus: send/receive the smaller blocks and receive/send the large one.
        // 2. Verify timestamp consistency:
        //   - if we split, then the block timestamps must be equal to `timestamp`.
        //   - if we merge, then `timestamp` must be at least each of the block timestamps.

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AccessAdapterCols<_, LEN, NUM_BLOCKS> = (*local).borrow();

        // 1. Memory bus interactions
        builder.assert_tern(local.dir + AB::F::ONE);
        let block_size = LEN / NUM_BLOCKS;
        assert_eq!(NUM_BLOCKS * block_size, LEN);
        for (i, (block, ts)) in local
            .values
            .chunks_exact(block_size)
            .zip(local.block_timestamps.iter())
            .enumerate()
        {
            self.memory_bus
                .send(
                    MemoryAddress::new(
                        local.addr_space,
                        local.pointer + AB::F::from_canonical_usize(i * block_size),
                    ),
                    block.to_vec(),
                    *ts,
                )
                .eval(builder, local.dir);
        }
        self.memory_bus
            .receive(
                MemoryAddress::new(local.addr_space, local.pointer),
                local.values.to_vec(),
                local.timestamp,
            )
            .eval(builder, local.dir);

        // 2. Timestamp consistency
        for (&ts, lt_aux) in local.block_timestamps.iter().zip(local.lt_aux.iter()) {
            // If split, then all block timestamps must be equal to `timestamp`.
            builder.assert_zero(local.dir * (local.dir + AB::F::ONE) * (local.timestamp - ts));
            // If merge, then `timestamp` must be at least each of the block timestamps.
            // In other words, `timestamp` must always be at least each of the block timestamps.
            // [AG] TODO: maybe it makes sense to only check this for merges after all?
            //            This will probably save tracegen time.
            self.lt_air.eval(
                builder,
                (
                    IsLessThanIo {
                        x: local.timestamp.into(),
                        y: ts.into(),
                        out: AB::Expr::ZERO,
                        count: AB::Expr::ONE,
                    },
                    lt_aux,
                ),
            );
        }
    }
}

enum GenericAccessAdapterChip {
    // [AG] TODO: un-hardcode 4
    N8(AccessAdapterChip<8, 4>),
    N16(AccessAdapterChip<16, 4>),
    N32(AccessAdapterChip<32, 4>),
}

impl GenericAccessAdapterChip {
    fn new<const N: usize>(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
    ) -> Self {
        let rc = range_checker;
        let mb = memory_bus;
        let cmb = clk_max_bits;
        match N {
            8 => GenericAccessAdapterChip::N8(AccessAdapterChip::new(rc, mb, cmb)),
            16 => GenericAccessAdapterChip::N16(AccessAdapterChip::new(rc, mb, cmb)),
            32 => GenericAccessAdapterChip::N32(AccessAdapterChip::new(rc, mb, cmb)),
            _ => panic!("Only supports N in (8, 16, 32)"),
        }
    }
}

pub struct AccessAdapterChip<const LEN: usize, const NUM_BLOCKS: usize> {
    air: AccessAdapterAir<LEN, NUM_BLOCKS>,
    range_checker: SharedVariableRangeCheckerChip,
    height: usize,
}

impl<const LEN: usize, const NUM_BLOCKS: usize> AccessAdapterChip<LEN, NUM_BLOCKS> {
    pub fn new(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
    ) -> Self {
        let lt_air = IsLtSubAir::new(range_checker.bus(), clk_max_bits);
        Self {
            air: AccessAdapterAir::<LEN, NUM_BLOCKS> { memory_bus, lt_air },
            range_checker,
            height: 0,
        }
    }

    pub fn execute_split<F: PrimeField32>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        timestamp: u32,
        row_slice: &mut [F],
    ) {
        let row: &mut AccessAdapterCols<F, LEN, NUM_BLOCKS> = row_slice.borrow_mut();
        row.addr_space = F::from_canonical_u32(address.address_space);
        row.pointer = F::from_canonical_u32(address.pointer);
        row.timestamp = F::from_canonical_u32(timestamp);
        row.dir = F::ONE;
        debug_assert_eq!(
            values.len(),
            LEN,
            "Input values slice length must match the fixed array length"
        );
        for (dest, src) in row.values.iter_mut().zip(values.iter()) {
            *dest = F::from_canonical_u8(*src);
        }
    }

    pub fn execute_merge<F: PrimeField32>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        block_timestamps: &[u32],
        timestamp: u32,
        row_slice: &mut [F],
    ) {
        let row: &mut AccessAdapterCols<F, LEN, NUM_BLOCKS> = row_slice.borrow_mut();
        row.addr_space = F::from_canonical_u32(address.address_space);
        row.pointer = F::from_canonical_u32(address.pointer);
        row.timestamp = F::from_canonical_u32(timestamp);
        row.dir = F::NEG_ONE;
        debug_assert_eq!(
            values.len(),
            LEN,
            "Input values slice length must match the fixed array length"
        );
        debug_assert_eq!(
            block_timestamps.len(),
            NUM_BLOCKS,
            "Input block timestamps slice length must match the number of blocks"
        );
        for (dest, src) in row.values.iter_mut().zip(values.iter()) {
            *dest = F::from_canonical_u8(*src);
        }
        for (dest, src) in row.block_timestamps.iter_mut().zip(block_timestamps.iter()) {
            *dest = F::from_canonical_u32(*src);
        }
    }

    pub fn fill_trace_row<F: PrimeField32>(&self, row_slice: &mut [F]) {
        let row: &mut AccessAdapterCols<F, LEN, NUM_BLOCKS> = row_slice.borrow_mut();
        if row.dir == F::ONE {
            row.block_timestamps.fill(row.timestamp);
        }
        // [AG] TODO: if we only want to do this for merges, we need to fix the `eval` method of the AIR.
        for (block_ts, lt_aux) in row.block_timestamps.iter().zip(row.lt_aux.iter_mut()) {
            let mut result = F::ZERO;
            self.air.lt_air.generate_subrow(
                (
                    self.range_checker.as_ref(),
                    F::as_canonical_u32(&row.timestamp),
                    F::as_canonical_u32(block_ts),
                ),
                (lt_aux, &mut result),
            );
        }
    }
}

pub struct AccessAdapterInventory {
    chips: Vec<GenericAccessAdapterChip>,
    air_names: Vec<String>,
}

impl AccessAdapterInventory {
    pub fn new(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
        max_access_adapter_n: usize,
    ) -> Self {
        let rc = range_checker;
        let mb = memory_bus;
        let cmb = clk_max_bits;
        let maan = max_access_adapter_n;
        assert!(matches!(maan, 8 | 16 | 32));
        let chips: Vec<_> = [
            Self::create_access_adapter_chip::<8>(rc.clone(), mb, cmb, maan),
            Self::create_access_adapter_chip::<16>(rc.clone(), mb, cmb, maan),
            Self::create_access_adapter_chip::<32>(rc.clone(), mb, cmb, maan),
        ]
        .into_iter()
        .flatten()
        .collect();
        let air_names = (0..chips.len())
            .map(|i| format!("AccessAdapter<{}>", 8 << i))
            .collect();
        Self { chips, air_names }
    }

    fn create_access_adapter_chip<const N: usize>(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
        max_access_adapter_n: usize,
    ) -> Option<GenericAccessAdapterChip> {
        if N <= max_access_adapter_n {
            Some(GenericAccessAdapterChip::new::<N>(
                range_checker,
                memory_bus,
                clk_max_bits,
            ))
        } else {
            None
        }
    }
}
