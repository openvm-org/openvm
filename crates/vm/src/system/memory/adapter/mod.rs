use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    ptr::copy_nonoverlapping,
    sync::Arc,
};

pub use air::*;
pub use columns::*;
use enum_dispatch::enum_dispatch;
use openvm_circuit_primitives::{
    is_less_than::IsLtSubAir, utils::next_power_of_two_or_zero,
    var_range::SharedVariableRangeCheckerChip, TraceSubRowGenerator,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::NATIVE_AS;
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_matrix::dense::RowMajorMatrix,
    p3_util::log2_strict_usize,
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};

use crate::{
    arch::{CustomBorrow, DenseRecordArena, RecordArena, SizedRecord},
    system::memory::{
        adapter::records::{
            size_by_layout, AccessLayout, AccessRecordHeader, AccessRecordMut,
            MERGE_AND_NOT_SPLIT_FLAG,
        },
        offline_checker::MemoryBus,
        MemoryAddress,
    },
};

mod air;
mod columns;
pub mod records;
#[cfg(test)]
mod tests;

pub struct AccessAdapterInventory<F> {
    chips: Vec<GenericAccessAdapterChip<F>>,
    pub arena: DenseRecordArena,
    air_names: Vec<String>,
}

impl<F: Clone + Send + Sync> AccessAdapterInventory<F> {
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
        assert!(matches!(maan, 2 | 4 | 8 | 16 | 32));
        let chips: Vec<_> = [
            Self::create_access_adapter_chip::<2>(rc.clone(), mb, cmb, maan),
            Self::create_access_adapter_chip::<4>(rc.clone(), mb, cmb, maan),
            Self::create_access_adapter_chip::<8>(rc.clone(), mb, cmb, maan),
            Self::create_access_adapter_chip::<16>(rc.clone(), mb, cmb, maan),
            Self::create_access_adapter_chip::<32>(rc.clone(), mb, cmb, maan),
        ]
        .into_iter()
        .flatten()
        .collect();
        let air_names = (0..chips.len()).map(|i| air_name(1 << (i + 1))).collect();
        Self {
            chips,
            arena: DenseRecordArena::with_capacity(0),
            air_names,
        }
    }

    pub fn num_access_adapters(&self) -> usize {
        self.chips.len()
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: Vec<usize>) {
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

    pub fn set_arena_from_trace_heights(&mut self, trace_heights: &[u32]) {
        assert_eq!(trace_heights.len(), self.chips.len());
        // At the very worst, each row in `Adapter<N>`
        // corresponds to a unique record of `block_size` being `2 * N`,
        // and its `lowest_block_size` is at least 1 and `type_size` is at most 4.
        let size_bound = trace_heights
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                size_by_layout(&AccessLayout {
                    block_size: 1 << (i + 1),
                    lowest_block_size: 1,
                    type_size: 4,
                }) * h as usize
            })
            .sum::<usize>();
        assert!(self
            .chips
            .iter()
            .all(|chip| chip.overridden_trace_height().is_none()));
        tracing::debug!(
            "Allocating {} bytes for memory adapters arena from heights {:?}",
            size_bound,
            trace_heights
        );
        self.arena.set_capacity(size_bound);
    }

    pub fn get_heights(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_height())
            .collect()
    }
    #[allow(dead_code)]
    pub fn get_widths(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip: &GenericAccessAdapterChip<F>| chip.trace_width())
            .collect()
    }
    pub fn get_cells(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_cells())
            .collect()
    }
    pub fn airs<SC: StarkGenericConfig>(&self) -> Vec<AirRef<SC>>
    where
        F: PrimeField32,
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.chips.iter().map(|chip| chip.air()).collect()
    }
    pub fn air_names(&self) -> Vec<String> {
        self.air_names.clone()
    }
    pub fn compute_trace_heights(&mut self) {
        let num_adapters = self.chips.len();
        let mut heights = vec![0; num_adapters];

        self.compute_heights_from_arena(&mut heights);
        self.apply_overridden_heights(&mut heights);
        for (chip, height) in self.chips.iter_mut().zip(heights) {
            chip.set_computed_trace_height(height);
        }
    }

    fn compute_heights_from_arena(&mut self, heights: &mut [usize]) {
        let bytes = self.arena.allocated_mut();
        tracing::debug!(
            "Computing heights from memory adapters arena: used {} bytes",
            bytes.len()
        );
        let mut ptr = 0;
        while ptr < bytes.len() {
            let header: &AccessRecordHeader = bytes[ptr..].borrow();
            let layout: AccessLayout = unsafe { bytes[ptr..].extract_layout() };
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

    pub fn generate_air_proof_inputs<SC: StarkGenericConfig>(mut self) -> Vec<AirProofInput<SC>>
    where
        F: PrimeField32,
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let num_adapters = self.chips.len();

        let mut heights = vec![0; num_adapters];
        self.compute_heights_from_arena(&mut heights);
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

        let mut trace_ptrs = vec![0; num_adapters];

        let bytes = self.arena.allocated_mut();
        let mut ptr = 0;
        while ptr < bytes.len() {
            let layout: AccessLayout = unsafe { bytes[ptr..].extract_layout() };
            let record: AccessRecordMut<'_> = bytes[ptr..].custom_borrow(layout.clone());
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
            .map(|trace| AirProofInput::simple_no_pis(trace))
            .collect()
    }

    fn create_access_adapter_chip<const N: usize>(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
        max_access_adapter_n: usize,
    ) -> Option<GenericAccessAdapterChip<F>>
    where
        F: Clone + Send + Sync,
    {
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

    pub(crate) fn alloc_record(&mut self, layout: AccessLayout) -> AccessRecordMut {
        self.arena.alloc(layout)
    }
}

#[enum_dispatch]
pub(crate) trait GenericAccessAdapterChipTrait<F> {
    fn set_override_trace_height(&mut self, overridden_height: usize);
    fn overridden_trace_height(&self) -> Option<usize>;
    fn set_computed_trace_height(&mut self, height: usize);

    fn fill_trace_row(
        &self,
        row: &mut [F],
        is_split: bool,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        left_timestamp: u32,
        right_timestamp: u32,
    ) where
        F: PrimeField32;
}

#[derive(Chip, ChipUsageGetter)]
#[enum_dispatch(GenericAccessAdapterChipTrait<F>)]
#[chip(where = "F: PrimeField32")]
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
        clk_max_bits: usize,
    ) -> Self {
        let rc = range_checker;
        let mb = memory_bus;
        let cmb = clk_max_bits;
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
    computed_trace_height: Option<usize>,
    _marker: PhantomData<F>,
}

impl<F: Clone + Send + Sync, const N: usize> AccessAdapterChip<F, N> {
    pub fn new(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
    ) -> Self {
        let lt_air = IsLtSubAir::new(range_checker.bus(), clk_max_bits);
        Self {
            air: AccessAdapterAir::<N> { memory_bus, lt_air },
            range_checker,
            overridden_height: None,
            computed_trace_height: None,
            _marker: PhantomData,
        }
    }
}
impl<F, const N: usize> GenericAccessAdapterChipTrait<F> for AccessAdapterChip<F, N> {
    fn set_override_trace_height(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }

    fn overridden_trace_height(&self) -> Option<usize> {
        self.overridden_height
    }

    fn set_computed_trace_height(&mut self, height: usize) {
        self.computed_trace_height = Some(height);
    }

    fn fill_trace_row(
        &self,
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
        // TODO: normal way
        if address.address_space < NATIVE_AS {
            for (dst, src) in row.values.iter_mut().zip(values.iter()) {
                *dst = F::from_canonical_u8(*src);
            }
        } else {
            unsafe {
                copy_nonoverlapping(
                    values.as_ptr(),
                    row.values.as_mut_ptr() as *mut u8,
                    N * size_of::<F>(),
                );
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

impl<SC: StarkGenericConfig, const N: usize> Chip<SC> for AccessAdapterChip<Val<SC>, N>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        unreachable!("AccessAdapterInventory should take care of adapters' trace generation")
    }
}

impl<F, const N: usize> ChipUsageGetter for AccessAdapterChip<F, N> {
    fn air_name(&self) -> String {
        air_name(N)
    }

    fn current_trace_height(&self) -> usize {
        self.computed_trace_height.unwrap_or(0)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

#[inline]
fn air_name(n: usize) -> String {
    format!("AccessAdapter<{}>", n)
}

#[inline(always)]
pub fn get_chip_index(block_size: usize) -> usize {
    assert!(
        block_size.is_power_of_two() && block_size >= 2,
        "Invalid block size {}",
        block_size
    );
    let index = block_size.trailing_zeros() - 1;
    index as usize
}
