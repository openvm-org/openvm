use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    ptr::copy_nonoverlapping,
    sync::Arc,
};

pub use air::*;
pub use columns::*;
use enum_dispatch::enum_dispatch;
use getset::Setters;
use openvm_circuit_primitives::{
    is_less_than::IsLtSubAir, utils::next_power_of_two_or_zero,
    var_range::SharedVariableRangeCheckerChip, TraceSubRowGenerator,
};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_air::BaseAir,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_matrix::dense::RowMajorMatrix,
    p3_util::log2_strict_usize,
    prover::{cpu::CpuBackend, types::AirProvingContext},
};

use crate::{
    arch::{
        AddressSpaceHostConfig, AddressSpaceHostLayout, CustomBorrow, DenseRecordArena,
        MemoryCellType, MemoryConfig, SizedRecord,
    },
    system::memory::{
        adapter::records::{
            arena_size_bound, AccessLayout, AccessRecordHeader, AccessRecordMut,
            MERGE_AND_NOT_SPLIT_FLAG,
        },
        offline_checker::MemoryBus,
        MemoryAddress,
    },
};

mod air;
mod columns;
pub mod records;

#[derive(Setters)]
pub struct AccessAdapterInventory<F> {
    pub(super) memory_config: MemoryConfig,
    chips: Vec<GenericAccessAdapterChip<F>>,
    #[getset(set = "pub")]
    arena: DenseRecordArena,
    #[cfg(feature = "metrics")]
    pub(crate) trace_heights: Vec<usize>,
}

impl<F: Clone + Send + Sync> AccessAdapterInventory<F> {
    pub fn new(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        memory_config: MemoryConfig,
    ) -> Self {
        let rc = range_checker;
        let mb = memory_bus;
        let tmb = memory_config.timestamp_max_bits;
        let maan = memory_config.max_access_adapter_n;
        assert!(matches!(maan, 2 | 4 | 8 | 16 | 32));
        let chips: Vec<_> = [
            Self::create_access_adapter_chip::<2>(rc.clone(), mb, tmb, maan),
            Self::create_access_adapter_chip::<4>(rc.clone(), mb, tmb, maan),
            Self::create_access_adapter_chip::<8>(rc.clone(), mb, tmb, maan),
            Self::create_access_adapter_chip::<16>(rc.clone(), mb, tmb, maan),
            Self::create_access_adapter_chip::<32>(rc.clone(), mb, tmb, maan),
        ]
        .into_iter()
        .flatten()
        .collect();
        Self {
            memory_config,
            chips,
            arena: DenseRecordArena::with_byte_capacity(0),
            #[cfg(feature = "metrics")]
            trace_heights: Vec::new(),
        }
    }

    pub fn num_access_adapters(&self) -> usize {
        self.chips.len()
    }

    pub(super) fn set_override_trace_heights(&mut self, overridden_heights: Vec<usize>) {
        self.set_arena_from_trace_heights(
            &overridden_heights
                .iter()
                .map(|&h| h as u32)
                .collect::<Vec<_>>(),
        );
        for (chip, oh) in self.chips.iter_mut().zip(overridden_heights) {
            chip.set_override_trace_height(oh);
        }
    }

    pub(super) fn set_arena_from_trace_heights(&mut self, trace_heights: &[u32]) {
        assert_eq!(trace_heights.len(), self.chips.len());
        let size_bound = arena_size_bound(trace_heights);
        tracing::debug!(
            "Allocating {} bytes for memory adapters arena from heights {:?}",
            size_bound,
            trace_heights
        );
        self.arena.set_byte_capacity(size_bound);
    }

    pub fn get_widths(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip: &GenericAccessAdapterChip<F>| chip.trace_width())
            .collect()
    }

    /// `heights` should have length equal to the number of access adapter chips.
    pub(crate) fn compute_heights_from_arena(arena: &DenseRecordArena, heights: &mut [usize]) {
        let bytes = arena.allocated();
        tracing::debug!(
            "Computing heights from memory adapters arena: used {} bytes",
            bytes.len()
        );
        let mut ptr = 0;
        while ptr < bytes.len() {
            let bytes_slice = &bytes[ptr..];
            let header: &AccessRecordHeader = bytes_slice.borrow();
            // SAFETY:
            // - bytes[ptr..] is a valid starting pointer to a previously allocated record
            // - The record contains self-describing layout information
            let layout: AccessLayout = unsafe { bytes_slice.extract_layout() };
            ptr += <AccessRecordMut<'_> as SizedRecord<AccessLayout>>::size(&layout);

            let log_max_block_size = log2_strict_usize(header.block_size as usize);
            for (i, h) in heights
                .iter_mut()
                .enumerate()
                .take(log_max_block_size)
                .skip(log2_strict_usize(header.lowest_block_size as usize))
            {
                *h += 1 << (log_max_block_size - i - 1);
            }
        }
        tracing::debug!("Computed heights from memory adapters arena: {:?}", heights);
    }

    fn apply_overridden_heights(&mut self, heights: &mut [usize]) {
        for (i, h) in heights.iter_mut().enumerate() {
            if let Some(oh) = self.chips[i].overridden_trace_height() {
                assert!(
                    oh >= *h,
                    "Overridden height {oh} is less than the required height {}",
                    *h
                );
                *h = oh;
            }
            *h = next_power_of_two_or_zero(*h);
        }
    }

    pub fn generate_proving_ctx<SC: StarkGenericConfig>(
        &mut self,
    ) -> Vec<AirProvingContext<CpuBackend<SC>>>
    where
        F: PrimeField32,
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let num_adapters = self.chips.len();

        let mut heights = vec![0; num_adapters];
        Self::compute_heights_from_arena(&self.arena, &mut heights);
        self.apply_overridden_heights(&mut heights);

        let widths = self
            .chips
            .iter()
            .map(|chip| chip.trace_width())
            .collect::<Vec<_>>();
        let mut traces = widths
            .iter()
            .zip(heights.iter())
            .map(|(&width, &height)| RowMajorMatrix::new(vec![F::ZERO; width * height], width))
            .collect::<Vec<_>>();
        #[cfg(feature = "metrics")]
        {
            self.trace_heights = heights;
        }

        let mut trace_ptrs = vec![0; num_adapters];

        let bytes = self.arena.allocated_mut();
        let mut ptr = 0;
        while ptr < bytes.len() {
            let bytes_slice = &mut bytes[ptr..];
            // SAFETY:
            // - bytes[ptr..] is a valid starting pointer to a previously allocated record
            // - The record contains self-describing layout information
            let layout: AccessLayout = unsafe { bytes_slice.extract_layout() };
            let record: AccessRecordMut<'_> = bytes_slice.custom_borrow(layout.clone());
            ptr += <AccessRecordMut<'_> as SizedRecord<AccessLayout>>::size(&layout);

            let log_min_block_size = log2_strict_usize(record.header.lowest_block_size as usize);
            let log_max_block_size = log2_strict_usize(record.header.block_size as usize);

            if record.header.timestamp_and_mask & MERGE_AND_NOT_SPLIT_FLAG != 0 {
                for i in log_min_block_size..log_max_block_size {
                    let data_len = layout.type_size << i;
                    let ts_len = 1 << (i - log_min_block_size);
                    for j in 0..record.data.len() / (2 * data_len) {
                        let row_slice =
                            &mut traces[i].values[trace_ptrs[i]..trace_ptrs[i] + widths[i]];
                        trace_ptrs[i] += widths[i];
                        self.chips[i].fill_trace_row(
                            &self.memory_config.addr_spaces,
                            row_slice,
                            false,
                            MemoryAddress::new(
                                record.header.address_space,
                                record.header.pointer + (j << (i + 1)) as u32,
                            ),
                            &record.data[j * 2 * data_len..(j + 1) * 2 * data_len],
                            *record.timestamps[2 * j * ts_len..(2 * j + 1) * ts_len]
                                .iter()
                                .max()
                                .unwrap(),
                            *record.timestamps[(2 * j + 1) * ts_len..(2 * j + 2) * ts_len]
                                .iter()
                                .max()
                                .unwrap(),
                        );
                    }
                }
            } else {
                let timestamp = record.header.timestamp_and_mask;
                for i in log_min_block_size..log_max_block_size {
                    let data_len = layout.type_size << i;
                    for j in 0..record.data.len() / (2 * data_len) {
                        let row_slice =
                            &mut traces[i].values[trace_ptrs[i]..trace_ptrs[i] + widths[i]];
                        trace_ptrs[i] += widths[i];
                        self.chips[i].fill_trace_row(
                            &self.memory_config.addr_spaces,
                            row_slice,
                            true,
                            MemoryAddress::new(
                                record.header.address_space,
                                record.header.pointer + (j << (i + 1)) as u32,
                            ),
                            &record.data[j * 2 * data_len..(j + 1) * 2 * data_len],
                            timestamp,
                            timestamp,
                        );
                    }
                }
            }
        }
        traces
            .into_iter()
            .map(|trace| AirProvingContext::simple_no_pis(Arc::new(trace)))
            .collect()
    }

    fn create_access_adapter_chip<const N: usize>(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        timestamp_max_bits: usize,
        max_access_adapter_n: usize,
    ) -> Option<GenericAccessAdapterChip<F>>
    where
        F: Clone + Send + Sync,
    {
        if N <= max_access_adapter_n {
            Some(GenericAccessAdapterChip::new::<N>(
                range_checker,
                memory_bus,
                timestamp_max_bits,
            ))
        } else {
            None
        }
    }
}

#[enum_dispatch]
pub(crate) trait GenericAccessAdapterChipTrait<F> {
    fn trace_width(&self) -> usize;
    fn set_override_trace_height(&mut self, overridden_height: usize);
    fn overridden_trace_height(&self) -> Option<usize>;

    #[allow(clippy::too_many_arguments)]
    fn fill_trace_row(
        &self,
        addr_spaces: &[AddressSpaceHostConfig],
        row: &mut [F],
        is_split: bool,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        left_timestamp: u32,
        right_timestamp: u32,
    ) where
        F: PrimeField32;
}

#[enum_dispatch(GenericAccessAdapterChipTrait<F>)]
enum GenericAccessAdapterChip<F> {
    N2(AccessAdapterChip<F, 2>),
    N4(AccessAdapterChip<F, 4>),
    N8(AccessAdapterChip<F, 8>),
    N16(AccessAdapterChip<F, 16>),
    N32(AccessAdapterChip<F, 32>),
}

impl<F: Clone + Send + Sync> GenericAccessAdapterChip<F> {
    fn new<const N: usize>(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        timestamp_max_bits: usize,
    ) -> Self {
        let rc = range_checker;
        let mb = memory_bus;
        let cmb = timestamp_max_bits;
        match N {
            2 => GenericAccessAdapterChip::N2(AccessAdapterChip::new(rc, mb, cmb)),
            4 => GenericAccessAdapterChip::N4(AccessAdapterChip::new(rc, mb, cmb)),
            8 => GenericAccessAdapterChip::N8(AccessAdapterChip::new(rc, mb, cmb)),
            16 => GenericAccessAdapterChip::N16(AccessAdapterChip::new(rc, mb, cmb)),
            32 => GenericAccessAdapterChip::N32(AccessAdapterChip::new(rc, mb, cmb)),
            _ => panic!("Only supports N in (2, 4, 8, 16, 32)"),
        }
    }
}

pub(crate) struct AccessAdapterChip<F, const N: usize> {
    air: AccessAdapterAir<N>,
    range_checker: SharedVariableRangeCheckerChip,
    overridden_height: Option<usize>,
    _marker: PhantomData<F>,
}

impl<F: Clone + Send + Sync, const N: usize> AccessAdapterChip<F, N> {
    pub fn new(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        timestamp_max_bits: usize,
    ) -> Self {
        let lt_air = IsLtSubAir::new(range_checker.bus(), timestamp_max_bits);
        Self {
            air: AccessAdapterAir::<N> { memory_bus, lt_air },
            range_checker,
            overridden_height: None,
            _marker: PhantomData,
        }
    }
}
impl<F, const N: usize> GenericAccessAdapterChipTrait<F> for AccessAdapterChip<F, N> {
    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }

    fn set_override_trace_height(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }

    fn overridden_trace_height(&self) -> Option<usize> {
        self.overridden_height
    }

    fn fill_trace_row(
        &self,
        addr_spaces: &[AddressSpaceHostConfig],
        row: &mut [F],
        is_split: bool,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        left_timestamp: u32,
        right_timestamp: u32,
    ) where
        F: PrimeField32,
    {
        let row: &mut AccessAdapterCols<F, N> = row.borrow_mut();
        row.is_valid = F::ONE;
        row.is_split = F::from_bool(is_split);
        row.address = MemoryAddress::new(
            F::from_canonical_u32(address.address_space),
            F::from_canonical_u32(address.pointer),
        );
        let addr_space_layout = addr_spaces[address.address_space as usize].layout;
        // SAFETY: values will be a slice of the cell type
        unsafe {
            match addr_space_layout {
                MemoryCellType::Native { .. } => {
                    copy_nonoverlapping(
                        values.as_ptr(),
                        row.values.as_mut_ptr() as *mut u8,
                        N * size_of::<F>(),
                    );
                }
                _ => {
                    for (dst, src) in row
                        .values
                        .iter_mut()
                        .zip(values.chunks_exact(addr_space_layout.size()))
                    {
                        *dst = addr_space_layout.to_field(src);
                    }
                }
            }
        }
        row.left_timestamp = F::from_canonical_u32(left_timestamp);
        row.right_timestamp = F::from_canonical_u32(right_timestamp);
        self.air.lt_air.generate_subrow(
            (self.range_checker.as_ref(), left_timestamp, right_timestamp),
            (&mut row.lt_aux, &mut row.is_right_larger),
        );
    }
}
