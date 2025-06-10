use std::{borrow::BorrowMut, io::Cursor, marker::PhantomData, sync::Arc};

pub use air::*;
pub use columns::*;
use enum_dispatch::enum_dispatch;
use openvm_circuit_primitives::{
    is_less_than::IsLtSubAir, utils::next_power_of_two_or_zero,
    var_range::SharedVariableRangeCheckerChip, TraceSubRowGenerator,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};

use crate::system::memory::{
    adapter::records::AccessAdapterRecordArena, offline_checker::MemoryBus, MemoryAddress,
};

mod air;
mod columns;
mod records;
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

pub struct AccessAdapterChip<F, const N: usize> {
    air: AccessAdapterAir<N>,
    range_checker: SharedVariableRangeCheckerChip,
    arena: AccessAdapterRecordArena,
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
            arena: AccessAdapterRecordArena::with_capacity(
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
        let width = self.trace_width();
        let records = self.arena.extract_bytes();
        let trace: Vec<F> = records
            .chunks_exact(size_of::<F>())
            .map(|chunk| unsafe { std::ptr::read(chunk.as_ptr() as *const F) })
            .collect();
        let mut trace = RowMajorMatrix::new(trace, width);
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
        // TODO(AG): everything related to the calculated trace height
        // needs to be in memory controller, who owns these traces.
    }

    fn execute_split(&mut self, address: MemoryAddress<u32, u32>, values: &[F], timestamp: u32)
    where
        F: PrimeField32,
    {
        let row_slice = self.arena.alloc_one::<AccessAdapterCols<F, N>>();
        let row: &mut AccessAdapterCols<F, N> = row_slice.borrow_mut();
        row.is_valid = F::ONE;
        row.is_split = F::ONE;
        row.address = MemoryAddress::new(
            F::from_canonical_u32(address.address_space),
            F::from_canonical_u32(address.pointer),
        );
        row.left_timestamp = F::from_canonical_u32(timestamp);
        row.right_timestamp = F::from_canonical_u32(timestamp);
        row.is_right_larger = F::ZERO;
        debug_assert_eq!(
            values.len(),
            N,
            "Input values slice length must match the access adapter type"
        );
        // TODO: move this to `fill_trace_row`
        self.air.lt_air.generate_subrow(
            (self.range_checker.as_ref(), timestamp, timestamp),
            (&mut row.lt_aux, &mut row.is_right_larger),
        );

        // SAFETY: `values` slice is asserted to have length N. `row.values` is an array of length
        // N. Pointers are valid and regions do not overlap because exactly one of them is a
        // part of the trace.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), row.values.as_mut_ptr(), N);
        }
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
        let row_slice = self.arena.alloc_one::<AccessAdapterCols<F, N>>();
        let row: &mut AccessAdapterCols<F, N> = row_slice.borrow_mut();
        row.is_valid = F::ONE;
        row.is_split = F::ZERO;
        row.address = MemoryAddress::new(
            F::from_canonical_u32(address.address_space),
            F::from_canonical_u32(address.pointer),
        );
        row.left_timestamp = F::from_canonical_u32(left_timestamp);
        row.right_timestamp = F::from_canonical_u32(right_timestamp);
        debug_assert_eq!(
            values.len(),
            N,
            "Input values slice length must match the access adapter type"
        );
        // TODO: move this to `fill_trace_row`
        self.air.lt_air.generate_subrow(
            (self.range_checker.as_ref(), left_timestamp, right_timestamp),
            (&mut row.lt_aux, &mut row.is_right_larger),
        );

        // SAFETY: `values` slice is asserted to have length N. `row.values` is an array of length
        // N. Pointers are valid and regions do not overlap because exactly one of them is a
        // part of the trace.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), row.values.as_mut_ptr(), N);
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
        self.arena.current_len() / self.trace_width() / size_of::<AccessAdapterCols<F, N>>()
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
