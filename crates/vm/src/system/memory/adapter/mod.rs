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
    arch::{CustomBorrow, DenseRecordArena, RecordArena, SizedRecord},
    system::memory::{
        adapter::records::{
            extract_metadata, size_by_layout, AccessLayout, AccessRecordMut, MERGE_BEFORE_FLAG,
            SPLIT_AFTER_FLAG,
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

    pub(crate) fn current_size(&self, block_size: usize) -> usize {
        self.chips[get_chip_index(block_size)].current_size()
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
        debug_assert!(self.is_marked_to_split(block_size, offset));
    }

    pub(crate) fn is_marked_to_split(&mut self, block_size: usize, offset: usize) -> bool {
        let index = get_chip_index(block_size);
        self.chips[index].is_marked_to_split(offset)
    }

    pub(crate) fn prepare_to_finalize(&mut self) {
        let mut chip_data: Vec<(
            &mut GenericAccessAdapterChip<F>,
            usize,
            Option<AccessLayout>,
        )> = self
            .chips
            .iter_mut()
            .map(|chip| {
                let ptr = 0;
                let layout = chip.extract_metadata_from(ptr);
                (chip, ptr, layout)
            })
            .collect();

        let extract_data = |chip: &mut GenericAccessAdapterChip<F>,
                            ptr: usize,
                            layout: &AccessLayout| {
            let record = chip.fancy_record_borrow_thing(ptr, layout.clone()).unwrap();
            let timestamp = (*record.timestamp_and_mask) & !SPLIT_AFTER_FLAG & !MERGE_BEFORE_FLAG;
            let address_space = *record.address_space;
            let pointer = *record.pointer;
            let block_size = *record.block_size;
            (timestamp, address_space, pointer, block_size)
        };

        loop {
            let next_data = chip_data
                .iter_mut()
                .flat_map(|(chip, ptr, layout)| {
                    layout
                        .as_mut()
                        .map(|layout| extract_data(chip, *ptr, layout))
                })
                .min();
            match next_data {
                None => break,
                Some(next_data) => {
                    let ids: Vec<usize> = chip_data
                        .iter_mut()
                        .enumerate()
                        .filter_map(|(i, (chip, ptr, layout))| match layout {
                            Some(layout) if extract_data(chip, *ptr, layout) == next_data => {
                                Some(i)
                            }
                            _ => None,
                        })
                        .collect();
                    let need_to_split = ids.iter().any(|&id| {
                        let (chip, ptr, layout) = &mut chip_data[id];
                        let record = chip
                            .fancy_record_borrow_thing(*ptr, layout.as_ref().unwrap().clone())
                            .unwrap();
                        *record.timestamp_and_mask & SPLIT_AFTER_FLAG != 0
                    });
                    if need_to_split {
                        for &id in ids.iter() {
                            let (chip, ptr, layout) = &mut chip_data[id];
                            let record = chip
                                .fancy_record_borrow_thing(*ptr, layout.as_ref().unwrap().clone())
                                .unwrap();
                            *record.timestamp_and_mask |= SPLIT_AFTER_FLAG;
                        }
                    }
                    for id in ids {
                        let (chip, ptr, layout) = &mut chip_data[id];
                        *ptr += size_by_layout(layout.as_ref().unwrap());
                        *layout = chip.extract_metadata_from(*ptr);
                    }
                }
            }
        }
    }
}

#[enum_dispatch]
pub(crate) trait GenericAccessAdapterChipTrait<F> {
    fn set_override_trace_heights(&mut self, overridden_height: usize);
    fn generate_trace(self) -> RowMajorMatrix<F>
    where
        F: PrimeField32;

    fn current_size(&self) -> usize;
    fn alloc_record(&mut self, layout: AccessLayout) -> AccessRecordMut;
    fn mark_to_split(&mut self, offset: usize);
    fn is_marked_to_split(&mut self, offset: usize) -> bool;
    fn extract_metadata_from(&self, offset: usize) -> Option<AccessLayout>;
    fn fancy_record_borrow_thing(
        &mut self,
        offset: usize,
        layout: AccessLayout,
    ) -> Option<AccessRecordMut<'_>>;
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
        let height = cur_height;
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
        let mut trace = RowMajorMatrix::new(vec![F::ZERO; width * padded_height], width);
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
                for i in 0..num_segs {
                    let row: &mut AccessAdapterCols<F, N> = trace.row_mut(row_idx).borrow_mut();
                    row.is_valid = F::ONE;
                    row.is_split = F::ZERO;
                    row.address = MemoryAddress::new(
                        F::from_canonical_u32(address_space),
                        F::from_canonical_u32(pointer + (i * N) as u32),
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
                    let left_timestamp = *timestamps[2 * i * timestamps.len() / (2 * num_segs)
                        ..((2 * i + 1) * timestamps.len()).div_ceil(2 * num_segs)]
                        .iter()
                        .max()
                        .unwrap();
                    let right_timestamp = *timestamps[(2 * i + 1) * timestamps.len()
                        / (2 * num_segs)
                        ..((2 * i + 2) * timestamps.len()).div_ceil(2 * num_segs)]
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
                        F::from_canonical_u32(pointer + (i * N) as u32),
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

        trace
    }

    fn current_size(&self) -> usize {
        self.arena.current_size()
    }

    fn alloc_record(&mut self, layout: AccessLayout) -> AccessRecordMut<'_> {
        self.arena.alloc(layout)
    }

    fn mark_to_split(&mut self, offset: usize) {
        let timestamp_and_mask = self.arena.transmute_from::<u32>(offset);
        *timestamp_and_mask |= SPLIT_AFTER_FLAG;
    }

    fn is_marked_to_split(&mut self, offset: usize) -> bool {
        let timestamp_and_mask = self.arena.transmute_from::<u32>(offset);
        *timestamp_and_mask & SPLIT_AFTER_FLAG != 0
    }

    fn extract_metadata_from(&self, offset: usize) -> Option<AccessLayout> {
        if offset < self.arena.records_buffer.position() as usize {
            Some(extract_metadata(
                &self.arena.records_buffer.get_ref()[offset..],
            ))
        } else {
            None
        }
    }

    fn fancy_record_borrow_thing(
        &mut self,
        offset: usize,
        layout: AccessLayout,
    ) -> Option<AccessRecordMut<'_>> {
        if offset < self.arena.records_buffer.position() as usize {
            Some(self.arena.records_buffer.get_mut()[offset..].custom_borrow(layout))
        } else {
            None
        }
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
        "Invalid block size {}",
        block_size
    );
    let index = block_size.trailing_zeros() - 1;
    index as usize
}
