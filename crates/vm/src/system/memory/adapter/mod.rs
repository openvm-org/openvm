use std::{borrow::BorrowMut, cmp::max, io::Cursor, sync::Arc};

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
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    p3_util::log2_strict_usize,
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};

use crate::system::memory::{offline_checker::MemoryBus, MemoryAddress};

mod air;
mod columns;
#[cfg(test)]
mod tests;

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

    pub fn num_access_adapters(&self) -> usize {
        self.chips.len()
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: Vec<usize>) {
        assert_eq!(overridden_heights.len(), self.chips.len());
        for (chip, oh) in self.chips.iter_mut().zip(overridden_heights) {
            chip.set_override_trace_heights(oh);
        }
    }

    // pub fn add_record(&mut self, record: AccessAdapterRecord<F>) {
    //     let n = record.data.len();
    //     let idx = log2_strict_usize(n) - 1;
    //     let chip = &mut self.chips[idx];
    //     debug_assert!(chip.n() == n);
    //     chip.add_record(record);
    // }

    // pub fn extend_records(&mut self, records: Vec<AccessAdapterRecord<F>>) {
    //     for record in records {
    //         self.add_record(record);
    //     }
    // }

    // #[cfg(test)]
    // pub fn records_for_n(&self, n: usize) -> &[AccessAdapterRecord<F>] {
    //     let idx = log2_strict_usize(n) - 1;
    //     let chip = &self.chips[idx];
    //     chip.records()
    // }

    // #[cfg(test)]
    // pub fn total_records(&self) -> usize {
    //     self.chips.iter().map(|chip| chip.records().len()).sum()
    // }

    pub fn get_heights(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_height())
            .collect()
    }

    // #[allow(dead_code)]
    // pub fn get_widths(&self) -> Vec<usize> {
    //     self.chips.iter().map(|chip| chip.trace_width()).collect()
    // }
    pub fn get_cells(&self) -> Vec<usize> {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_cells())
            .collect()
    }

    pub fn airs<F: PrimeField32, SC: StarkGenericConfig>(&self) -> Vec<AirRef<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.chips.iter().map(|chip| chip.air()).collect()
    }

    pub fn air_names(&self) -> Vec<String> {
        self.air_names.clone()
    }

    pub fn generate_air_proof_inputs<F: PrimeField32, SC: StarkGenericConfig>(
        self,
    ) -> Vec<AirProofInput<SC>>
    where
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

    /// Executes a split operation for the given block size `N`.
    /// `N` refers to the size of the block *before* splitting.
    ///
    /// # Panics
    /// Panics if `N` is invalid (not 8, 16, or 32, or not a power of two)
    /// or if the corresponding chip is not available (due to `max_access_adapter_n`).
    pub fn execute_split<const N: usize, F: PrimeField32>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        timestamp: u32,
        row_slice: &mut [F],
    ) {
        let index = get_chip_index::<N>();
        let chip = self
            .chips
            .get_mut(index)
            .unwrap_or_else(|| panic!("AccessAdapterChip for block size {} not found", N));
        chip.execute_split(address, values, timestamp, row_slice);
    }

    /// Executes a merge operation for the given block size `N`.
    /// `N` refers to the size of the block *after* merging.
    ///
    /// # Panics
    /// Panics if `N` is invalid (not 8, 16, or 32, or not a power of two)
    /// or if the corresponding chip is not available (due to `max_access_adapter_n`).
    pub fn execute_merge<const N: usize, F: PrimeField32>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        left_timestamp: u32,
        right_timestamp: u32,
        row_slice: &mut [F],
    ) {
        let index = get_chip_index::<N>();
        let chip = self
            .chips
            .get_mut(index)
            .unwrap_or_else(|| panic!("AccessAdapterChip for block size {} not found", N));
        chip.execute_merge(address, values, left_timestamp, right_timestamp, row_slice);
    }

    /// Fills the trace row for a split/merge operation related to the given block size `N`.
    /// `N` refers to the size of the larger block involved in the operation.
    ///
    /// # Panics
    /// Panics if `N` is invalid (not 8, 16, or 32, or not a power of two)
    /// or if the corresponding chip is not available (due to `max_access_adapter_n`).
    pub fn fill_trace_row<const N: usize, F: PrimeField32>(&mut self, row_slice: &mut [F]) {
        let index = get_chip_index::<N>();
        let chip = self
            .chips
            .get_mut(index)
            .unwrap_or_else(|| panic!("AccessAdapterChip for block size {} not found", N));
        // Assuming PrimeField32 and TraceRow are in scope
        chip.fill_trace_row::<F>(row_slice);
    }
}

#[derive(ChipUsageGetter)]
enum GenericAccessAdapterChip {
    N2(AccessAdapterChip<2>),
    N4(AccessAdapterChip<4>),
    N8(AccessAdapterChip<8>),
    N16(AccessAdapterChip<16>),
    N32(AccessAdapterChip<32>),
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
            2 => GenericAccessAdapterChip::N2(AccessAdapterChip::new(rc, mb, cmb)),
            4 => GenericAccessAdapterChip::N4(AccessAdapterChip::new(rc, mb, cmb)),
            8 => GenericAccessAdapterChip::N8(AccessAdapterChip::new(rc, mb, cmb)),
            16 => GenericAccessAdapterChip::N16(AccessAdapterChip::new(rc, mb, cmb)),
            32 => GenericAccessAdapterChip::N32(AccessAdapterChip::new(rc, mb, cmb)),
            _ => panic!("Only supports N in (8, 16, 32)"),
        }
    }

    /// Dispatches the execute_split call to the appropriate concrete chip.
    /// Assumes the concrete chip has a method `execute_split(address_space: u32, pointer: u32)`.
    fn execute_split<F: PrimeField32>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        timestamp: u32,
        row_slice: &mut [F],
    ) {
        match self {
            GenericAccessAdapterChip::N2(chip) => {
                chip.execute_split(address, values, timestamp, row_slice)
            }
            GenericAccessAdapterChip::N4(chip) => {
                chip.execute_split(address, values, timestamp, row_slice)
            }
            GenericAccessAdapterChip::N8(chip) => {
                chip.execute_split(address, values, timestamp, row_slice)
            }
            GenericAccessAdapterChip::N16(chip) => {
                chip.execute_split(address, values, timestamp, row_slice)
            }
            GenericAccessAdapterChip::N32(chip) => {
                chip.execute_split(address, values, timestamp, row_slice)
            }
        }
    }

    /// Dispatches the execute_merge call to the appropriate concrete chip.
    /// Assumes the concrete chip has a method `execute_merge(address_space: u32, pointer: u32)`.
    fn execute_merge<F: PrimeField32>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[u8],
        left_timestamp: u32,
        right_timestamp: u32,
        row_slice: &mut [F],
    ) {
        match self {
            GenericAccessAdapterChip::N2(chip) => {
                chip.execute_merge(address, values, left_timestamp, right_timestamp, row_slice)
            }
            GenericAccessAdapterChip::N4(chip) => {
                chip.execute_merge(address, values, left_timestamp, right_timestamp, row_slice)
            }
            GenericAccessAdapterChip::N8(chip) => {
                chip.execute_merge(address, values, left_timestamp, right_timestamp, row_slice)
            }
            GenericAccessAdapterChip::N16(chip) => {
                chip.execute_merge(address, values, left_timestamp, right_timestamp, row_slice)
            }
            GenericAccessAdapterChip::N32(chip) => {
                chip.execute_merge(address, values, left_timestamp, right_timestamp, row_slice)
            }
        }
    }

    /// Dispatches the fill_trace_row call to the appropriate concrete chip.
    /// Assumes the concrete chip has a method `fill_trace_row<F>(row: &mut TraceRow<F>,
    /// address_space: u32, pointer: u32)`. Also assumes `TraceRow<F>` is the correct type as
    /// used by the caller (`AccessAdapterInventory`).
    fn fill_trace_row<F: PrimeField32>(&mut self, row_slice: &mut [F]) {
        match self {
            GenericAccessAdapterChip::N2(chip) => chip.fill_trace_row::<F>(row_slice),
            GenericAccessAdapterChip::N4(chip) => chip.fill_trace_row::<F>(row_slice),
            GenericAccessAdapterChip::N8(chip) => chip.fill_trace_row::<F>(row_slice),
            GenericAccessAdapterChip::N16(chip) => chip.fill_trace_row::<F>(row_slice),
            GenericAccessAdapterChip::N32(chip) => chip.fill_trace_row::<F>(row_slice),
        }
    }

    fn set_override_trace_heights(&mut self, _height: usize) {}

    fn air<SC: StarkGenericConfig>(&self) -> AirRef<SC> {
        match self {
            GenericAccessAdapterChip::N2(chip) => chip.air(),
            GenericAccessAdapterChip::N4(chip) => chip.air(),
            GenericAccessAdapterChip::N8(chip) => chip.air(),
            GenericAccessAdapterChip::N16(chip) => chip.air(),
            GenericAccessAdapterChip::N32(chip) => chip.air(),
        }
    }

    fn generate_air_proof_input<SC: StarkGenericConfig>(self) -> AirProofInput<SC> {
        todo!()
    }
}

pub struct AccessAdapterChip<const LEN: usize> {
    air: AccessAdapterAir<LEN>,
    range_checker: SharedVariableRangeCheckerChip,
    height: usize,
}

impl<const LEN: usize> AccessAdapterChip<LEN> {
    pub fn new(
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        clk_max_bits: usize,
    ) -> Self {
        let lt_air = IsLtSubAir::new(range_checker.bus(), clk_max_bits);
        Self {
            air: AccessAdapterAir::<LEN> { memory_bus, lt_air },
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
        let row: &mut AccessAdapterCols<F, LEN> = row_slice.borrow_mut();
        row.is_valid = F::ONE;
        row.is_split = F::ONE;
        row.address = MemoryAddress::new(
            F::from_canonical_u32(address.address_space),
            F::from_canonical_u32(address.pointer),
        );
        let timestamp = F::from_canonical_u32(timestamp);
        row.left_timestamp = timestamp;
        row.right_timestamp = timestamp;
        row.is_right_larger = F::ZERO;
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
        left_timestamp: u32,
        right_timestamp: u32,
        row_slice: &mut [F],
    ) {
        let row: &mut AccessAdapterCols<F, LEN> = row_slice.borrow_mut();
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
            LEN,
            "Input values slice length must match the fixed array length"
        );
        for (dest, src) in row.values.iter_mut().zip(values.iter()) {
            *dest = F::from_canonical_u8(*src);
        }
    }

    pub fn fill_trace_row<F: PrimeField32>(&self, row_slice: &mut [F]) {
        let row: &mut AccessAdapterCols<F, LEN> = row_slice.borrow_mut();
        self.air.lt_air.generate_subrow(
            (
                self.range_checker.as_ref(),
                F::as_canonical_u32(&row.left_timestamp),
                F::as_canonical_u32(&row.right_timestamp),
            ),
            (&mut row.lt_aux, &mut row.is_right_larger),
        );
    }

    fn set_override_trace_heights(&mut self, _height: usize) {}
    fn air<SC: StarkGenericConfig>(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }
    fn generate_air_proof_input<SC: StarkGenericConfig>(self) -> AirProofInput<SC> {
        todo!()
    }
}

// impl<SC: StarkGenericConfig, const N: usize> Chip<SC> for AccessAdapterChip<Val<SC>, N>
// where
//     Val<SC>: PrimeField32,
// {
//     fn air(&self) -> AirRef<SC> {
//         Arc::new(self.air.clone())
//     }

//     fn generate_air_proof_input(self) -> AirProofInput<SC> {
//         let trace = self.generate_trace();
//         AirProofInput::simple_no_pis(trace)
//     }
// }

impl<const N: usize> ChipUsageGetter for AccessAdapterChip<N> {
    fn air_name(&self) -> String {
        air_name(N)
    }

    fn current_trace_height(&self) -> usize {
        // TODO: get rid of this
        0
    }

    fn trace_width(&self) -> usize {
        BaseAir::<u8>::width(&self.air)
    }
}

#[inline]
fn air_name(n: usize) -> String {
    format!("AccessAdapter<{}>", n)
}

#[inline(always)]
fn get_chip_index<const N: usize>() -> usize {
    assert!(
        N.is_power_of_two() && N >= 2,
        "Invalid block size {} for split operation",
        N
    );
    // N.trailing_zeros() gives log2(N)
    let index = N.trailing_zeros().checked_sub(1).unwrap();
    index as usize
}

pub struct AdapterInventoryTraceCursor<F> {
    // [AG] TODO: replace with a pre-allocated space
    cursors: Vec<Cursor<Vec<F>>>,
}

impl<F: PrimeField32> AdapterInventoryTraceCursor<F> {
    pub fn new(as_cnt: usize) -> Self {
        let cursors = vec![Cursor::new(Vec::new()); as_cnt];
        Self { cursors }
    }

    pub fn get_row_slice<const N: usize>(&mut self) -> &mut [F] {
        let index = get_chip_index::<N>();
        let begin = self.cursors[index].position() as usize;
        let end = begin + size_of::<AccessAdapterCols<u8, N>>();
        self.cursors[index].set_position(end as u64);
        &mut self.cursors[index].get_mut()[begin..end]
    }
}
