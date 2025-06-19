use std::{borrow::BorrowMut, marker::PhantomData, sync::Arc};

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
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};

use crate::{
    arch::{DenseRecordArena, RecordArena, SizedRecord},
    system::memory::{
        adapter::records::{
            extract_metadata, fancy_record_borrow_thing, AccessLayout, AccessRecordMut,
            MERGE_BEFORE_FLAG, SPLIT_AFTER_FLAG,
        },
        offline_checker::MemoryBus,
        MemoryAddress,
    },
};

mod air;
mod columns;
pub(crate) mod records;
#[cfg(test)]
mod tests;

pub struct AccessAdapterInventory<F> {
    chips: Vec<GenericAccessAdapterChip<F>>,
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
        Self { chips, air_names }
    }
    pub fn num_access_adapters(&self) -> usize {
        self.chips.len()
    }
    pub fn set_override_trace_heights(&mut self, overridden_heights: Vec<usize>) {
        assert_eq!(overridden_heights.len(), self.chips.len());
        for (chip, oh) in self.chips.iter_mut().zip(overridden_heights) {
            chip.set_override_trace_heights(oh);
        }
    }

    pub fn get_heights(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_height())
            .collect()
    }
    #[allow(dead_code)]
    pub fn get_widths(&self) -> Vec<usize> {
        self.chips.iter().map(|chip| chip.trace_width()).collect()
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
    pub fn generate_air_proof_inputs<SC: StarkGenericConfig>(self) -> Vec<AirProofInput<SC>>
    where
        F: PrimeField32,
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.chips
            .into_iter()
            .map(|chip| chip.generate_air_proof_input())
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

    pub(crate) fn current_size<const N: usize>(&self) -> usize {
        self.chips[get_chip_index(N)].current_size()
    }

    pub(crate) fn alloc_record(
        &mut self,
        block_size: usize,
        layout: AccessLayout,
    ) -> AccessRecordMut {
        let index = get_chip_index(block_size);
        self.chips[index].alloc_record(layout)
    }

    pub(crate) fn mark_to_split(&mut self, block_size: usize, offset: usize) {
        let index = get_chip_index(block_size);
        self.chips[index].mark_to_split(offset);
    }

    pub(crate) fn execute_split(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        timestamp: u32,
    ) where
        F: PrimeField32,
    {
        let index = get_chip_index(values.len());
        self.chips[index].execute_split(address, values, timestamp);
    }

    pub(crate) fn execute_merge(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        left_timestamp: u32,
        right_timestamp: u32,
    ) where
        F: PrimeField32,
    {
        let index = get_chip_index(values.len());
        self.chips[index].execute_merge(address, values, left_timestamp, right_timestamp);
    }
}

#[enum_dispatch]
pub trait GenericAccessAdapterChipTrait<F> {
    fn set_override_trace_heights(&mut self, overridden_height: usize);
    fn n(&self) -> usize;
    fn generate_trace(self) -> RowMajorMatrix<F>
    where
        F: PrimeField32;

    fn current_size(&self) -> usize;
    fn alloc_record(&mut self, layout: AccessLayout) -> AccessRecordMut;
    fn mark_to_split(&mut self, offset: usize);

    fn execute_split(&mut self, address: MemoryAddress<u32, u32>, values: &[F], timestamp: u32)
    where
        F: PrimeField32;

    fn execute_merge(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
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
    arena: DenseRecordArena,
    overridden_height: Option<usize>,
    _marker: PhantomData<F>,
}

const MAX_ARENA_SIZE: usize = 1 << 22;

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
            arena: DenseRecordArena::with_capacity(
                MAX_ARENA_SIZE * size_of::<AccessAdapterCols<F, N>>(),
            ),
            overridden_height: None,
            _marker: PhantomData,
        }
    }
}
impl<F, const N: usize> GenericAccessAdapterChipTrait<F> for AccessAdapterChip<F, N> {
    fn set_override_trace_heights(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }
    fn n(&self) -> usize {
        N
    }
    fn generate_trace(self) -> RowMajorMatrix<F>
    where
        F: PrimeField32,
    {
        let mut milestones = vec![(0, 0)];
        // TODO(AG): replace all this with nice records reading function we will have,
        // or even move it to GPU idk
        let mut cur_height = 0;
        let bytes = self.arena.allocated();
        let mut ptr = 0;
        while ptr < bytes.len() {
            let timestamp_and_mask =
                unsafe { std::ptr::read(bytes.as_ptr().add(ptr) as *const u32) };
            let num_rows = if timestamp_and_mask & SPLIT_AFTER_FLAG != 0 {
                1
            } else {
                0
            } + if timestamp_and_mask & MERGE_BEFORE_FLAG != 0 {
                1
            } else {
                0
            };
            let layout = extract_metadata(&bytes[ptr..]);
            let num_rows = num_rows * (layout.block_size / N);
            ptr += SizedRecord::<AccessLayout, AccessRecordMut<'_>>::size(&self.arena, &layout);

            cur_height += num_rows;
            milestones.push((ptr, cur_height));
        }

        let width = self.trace_width();
        let mut trace = RowMajorMatrix::new(vec![F::ZERO; width * cur_height], width);
        for (&(start_ptr, start_height), &(_, _)) in
            milestones.iter().zip(milestones.iter().skip(1))
        {
            let mut ptr = start_ptr;
            let layout = extract_metadata(&bytes[start_ptr..]);

            // TODO(AG): get rid of this

            // timestamp_and_mask: u32 (4 bytes)
            let timestamp_and_mask = unsafe { *(bytes.as_ptr().add(ptr) as *const u32) };
            ptr += 4;

            // address_space: u32 (4 bytes)
            let address_space = unsafe { *(bytes.as_ptr().add(ptr) as *const u32) };
            ptr += 4;

            // pointer: u32 (4 bytes)
            let pointer = unsafe { *(bytes.as_ptr().add(ptr) as *const u32) };
            ptr += 4;

            // block_size: u32 (4 bytes)
            let _block_size_field = unsafe { *(bytes.as_ptr().add(ptr) as *const u32) };
            ptr += 4;

            // data: [u8] (block_size * type_size bytes)
            let data = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr().add(ptr),
                    layout.block_size * layout.type_size,
                )
            };
            ptr += layout.block_size * layout.type_size;

            // prev_data: [u8] (block_size * type_size bytes)
            let prev_data = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr().add(ptr),
                    layout.block_size * layout.type_size,
                )
            };
            ptr += layout.block_size * layout.type_size;

            // timestamps: [u32] (block_size / cell_size * 4 bytes)
            let timestamps = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr().add(ptr) as *const u32,
                    layout.block_size / layout.cell_size,
                )
            };

            let mut row_idx = start_height;
            let num_segs = layout.block_size / N;
            if timestamp_and_mask & MERGE_BEFORE_FLAG != 0 {
                let timestamps_batch_len = N / 2;
                for i in 0..num_segs {
                    let row: &mut AccessAdapterCols<F, N> = trace.row_mut(row_idx).borrow_mut();
                    row.is_valid = F::ONE;
                    row.is_split = F::ZERO;
                    row.address = MemoryAddress::new(
                        F::from_canonical_u32(address_space),
                        F::from_canonical_u32(pointer),
                    );
                    if address_space < NATIVE_AS {
                        for j in 0..N {
                            row.values[j] = F::from_canonical_u8(prev_data[i * N + j]);
                        }
                    } else {
                        for j in 0..N {
                            row.values[j] = unsafe {
                                std::ptr::read(
                                    prev_data.as_ptr().add((i * N + j) * layout.type_size)
                                        as *const F,
                                )
                            };
                        }
                    }
                    let left_timestamp = *timestamps
                        [2 * i * timestamps_batch_len..(2 * i + 1) * timestamps_batch_len]
                        .iter()
                        .max()
                        .unwrap();
                    let right_timestamp = *timestamps
                        [(2 * i + 1) * timestamps_batch_len..(2 * i + 2) * timestamps_batch_len]
                        .iter()
                        .max()
                        .unwrap();
                    row.left_timestamp = F::from_canonical_u32(left_timestamp);
                    row.right_timestamp = F::from_canonical_u32(right_timestamp);
                    self.air.lt_air.generate_subrow(
                        (self.range_checker.as_ref(), left_timestamp, right_timestamp),
                        (&mut row.lt_aux, &mut row.is_right_larger),
                    );

                    row_idx += 1;
                }
            }
            if timestamp_and_mask & SPLIT_AFTER_FLAG != 0 {
                for i in 0..num_segs {
                    let row: &mut AccessAdapterCols<F, N> = trace.row_mut(row_idx).borrow_mut();
                    row.is_valid = F::ONE;
                    row.is_split = F::ONE;
                    row.address = MemoryAddress::new(
                        F::from_canonical_u32(address_space),
                        F::from_canonical_u32(pointer),
                    );
                    if address_space < NATIVE_AS {
                        for j in 0..N {
                            row.values[j] = F::from_canonical_u8(data[i * N + j]);
                        }
                    } else {
                        for j in 0..N {
                            row.values[j] = unsafe {
                                std::ptr::read(
                                    data.as_ptr().add((i * N + j) * layout.type_size) as *const F
                                )
                            };
                        }
                    }
                    let timestamp = timestamp_and_mask & !SPLIT_AFTER_FLAG & !MERGE_BEFORE_FLAG;
                    row.left_timestamp = F::from_canonical_u32(timestamp);
                    row.right_timestamp = F::from_canonical_u32(timestamp);
                    self.air.lt_air.generate_subrow(
                        (self.range_checker.as_ref(), timestamp, timestamp),
                        (&mut row.lt_aux, &mut row.is_right_larger),
                    );

                    row_idx += 1;
                }
            }
        }

        let height = trace.height();
        let padded_height = if let Some(oh) = self.overridden_height {
            assert!(
                oh >= height,
                "Overridden height {oh} is less than the required height {height}"
            );
            oh
        } else {
            height
        };
        let padded_height = next_power_of_two_or_zero(padded_height);
        trace.pad_to_height(padded_height, F::ZERO);
        trace
    }

    fn current_size(&self) -> usize {
        self.arena.current_size()
    }

    fn alloc_record(&mut self, layout: AccessLayout) -> AccessRecordMut<'_> {
        self.arena.alloc(layout)
    }

    fn mark_to_split(&mut self, offset: usize) {
        let timestamp_and_mask = self.arena.transmute_from::<'_, u32>(offset);
        *timestamp_and_mask |= SPLIT_AFTER_FLAG;
    }

    fn execute_split(&mut self, address: MemoryAddress<u32, u32>, values: &[F], timestamp: u32)
    where
        F: PrimeField32,
    {
        todo!()
        // let row_slice = self.arena.alloc_one::<AccessAdapterCols<F, N>>();
        // let row: &mut AccessAdapterCols<F, N> = row_slice.borrow_mut();
        // row.is_valid = F::ONE;
        // row.is_split = F::ONE;
        // row.address = MemoryAddress::new(
        //     F::from_canonical_u32(address.address_space),
        //     F::from_canonical_u32(address.pointer),
        // );
        // row.left_timestamp = F::from_canonical_u32(timestamp);
        // row.right_timestamp = F::from_canonical_u32(timestamp);
        // row.is_right_larger = F::ZERO;
        // debug_assert_eq!(
        //     values.len(),
        //     N,
        //     "Input values slice length must match the access adapter type"
        // );
        // // TODO: move this to `fill_trace_row`
        // self.air.lt_air.generate_subrow(
        //     (self.range_checker.as_ref(), timestamp, timestamp),
        //     (&mut row.lt_aux, &mut row.is_right_larger),
        // );

        // // SAFETY: `values` slice is asserted to have length N. `row.values` is an array of
        // length // N. Pointers are valid and regions do not overlap because exactly one of
        // them is a // part of the trace.
        // unsafe {
        //     std::ptr::copy_nonoverlapping(values.as_ptr(), row.values.as_mut_ptr(), N);
        // }
    }

    fn execute_merge(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        left_timestamp: u32,
        right_timestamp: u32,
    ) where
        F: PrimeField32,
    {
        todo!()
        // let row_slice = self.arena.alloc_one::<AccessAdapterCols<F, N>>();
        // let row: &mut AccessAdapterCols<F, N> = row_slice.borrow_mut();
        // row.is_valid = F::ONE;
        // row.is_split = F::ZERO;
        // row.address = MemoryAddress::new(
        //     F::from_canonical_u32(address.address_space),
        //     F::from_canonical_u32(address.pointer),
        // );
        // row.left_timestamp = F::from_canonical_u32(left_timestamp);
        // row.right_timestamp = F::from_canonical_u32(right_timestamp);
        // debug_assert_eq!(
        //     values.len(),
        //     N,
        //     "Input values slice length must match the access adapter type"
        // );
        // // TODO: move this to `fill_trace_row`
        // self.air.lt_air.generate_subrow(
        //     (self.range_checker.as_ref(), left_timestamp, right_timestamp),
        //     (&mut row.lt_aux, &mut row.is_right_larger),
        // );

        // // SAFETY: `values` slice is asserted to have length N. `row.values` is an array of
        // length // N. Pointers are valid and regions do not overlap because exactly one of
        // them is a // part of the trace.
        // unsafe {
        //     std::ptr::copy_nonoverlapping(values.as_ptr(), row.values.as_mut_ptr(), N);
        // }
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
        let trace = self.generate_trace();
        AirProofInput::simple_no_pis(trace)
    }
}

impl<F, const N: usize> ChipUsageGetter for AccessAdapterChip<F, N> {
    fn air_name(&self) -> String {
        air_name(N)
    }

    fn current_trace_height(&self) -> usize {
        self.arena.records_buffer.position() as usize / size_of::<AccessAdapterCols<F, N>>()
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
        "Invalid block size {} for split operation",
        block_size
    );
    let index = block_size.trailing_zeros() - 1;
    index as usize
}
