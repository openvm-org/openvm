use std::{borrow::BorrowMut, io::Cursor, sync::Arc};

pub use air::*;
pub use columns::*;
use enum_dispatch::enum_dispatch;
use openvm_circuit_primitives::{
    is_less_than::IsLtSubAir, var_range::SharedVariableRangeCheckerChip, TraceSubRowGenerator,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_util::log2_strict_usize,
    prover::types::AirProofInput,
    AirRef, Chip, ChipUsageGetter,
};

use crate::system::memory::{offline_checker::MemoryBus, MemoryAddress};

mod air;
mod columns;
#[cfg(test)]
mod tests;

pub struct AccessAdapterInventory<F> {
    pub chips: Vec<GenericAccessAdapterChip<F>>,
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
    pub fn add_record(&mut self, record: AccessAdapterRecord<F>) {
        let n = record.data.len();
        let idx = log2_strict_usize(n) - 1;
        let chip = &mut self.chips[idx];
        debug_assert!(chip.n() == n);
        chip.add_record(record);
    }

    pub fn extend_records(&mut self, records: Vec<AccessAdapterRecord<F>>) {
        for record in records {
            self.add_record(record);
        }
    }

    pub fn set_trace(&mut self, index: usize, trace: Vec<F>, width: usize)
    where
        F: PrimeField32,
    {
        let trace = RowMajorMatrix::new(trace, width);
        self.chips[index].set_trace(trace);
    }

    #[cfg(test)]
    pub fn records_for_n(&self, n: usize) -> &[AccessAdapterRecord<F>] {
        let idx = log2_strict_usize(n) - 1;
        let chip = &self.chips[idx];
        chip.records()
    }

    #[cfg(test)]
    pub fn total_records(&self) -> usize {
        self.chips.iter().map(|chip| chip.records().len()).sum()
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
        row_slice: &mut [F],
    ) where
        F: PrimeField32,
    {
        let index = get_chip_index(values.len());
        self.chips[index].execute_split(address, values, timestamp, row_slice);
    }

    pub(crate) fn execute_merge(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        left_timestamp: u32,
        right_timestamp: u32,
        row_slice: &mut [F],
    ) where
        F: PrimeField32,
    {
        let index = get_chip_index(values.len());
        self.chips[index].execute_merge(
            address,
            values,
            left_timestamp,
            right_timestamp,
            row_slice,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessAdapterRecordKind {
    Split,
    Merge {
        left_timestamp: u32,
        right_timestamp: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessAdapterRecord<T> {
    pub timestamp: u32,
    pub address_space: T,
    pub start_index: T,
    pub data: Vec<T>,
    pub kind: AccessAdapterRecordKind,
}

#[enum_dispatch]
pub trait GenericAccessAdapterChipTrait<F> {
    fn set_override_trace_heights(&mut self, overridden_height: usize);
    fn add_record(&mut self, record: AccessAdapterRecord<F>);
    fn n(&self) -> usize;
    fn generate_trace(self) -> RowMajorMatrix<F>
    where
        F: PrimeField32;
    fn set_trace(&mut self, trace: RowMajorMatrix<F>)
    where
        F: PrimeField32;

    fn execute_split(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        timestamp: u32,
        row_slice: &mut [F],
    ) where
        F: PrimeField32;

    fn execute_merge(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        left_timestamp: u32,
        right_timestamp: u32,
        row_slice: &mut [F],
    ) where
        F: PrimeField32;
}

#[derive(Chip, ChipUsageGetter)]
#[enum_dispatch(GenericAccessAdapterChipTrait<F>)]
#[chip(where = "F: PrimeField32")]
pub enum GenericAccessAdapterChip<F> {
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

    #[cfg(test)]
    fn records(&self) -> &[AccessAdapterRecord<F>] {
        match &self {
            GenericAccessAdapterChip::N2(chip) => &chip.records,
            GenericAccessAdapterChip::N4(chip) => &chip.records,
            GenericAccessAdapterChip::N8(chip) => &chip.records,
            GenericAccessAdapterChip::N16(chip) => &chip.records,
            GenericAccessAdapterChip::N32(chip) => &chip.records,
        }
    }
}

pub struct AccessAdapterChip<F, const N: usize> {
    air: AccessAdapterAir<N>,
    range_checker: SharedVariableRangeCheckerChip,
    pub records: Vec<AccessAdapterRecord<F>>,
    pub trace: RowMajorMatrix<F>,
    overridden_height: Option<usize>,
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
            records: vec![],
            trace: RowMajorMatrix::new(Vec::new(), 0),
            overridden_height: None,
        }
    }
}
impl<F, const N: usize> GenericAccessAdapterChipTrait<F> for AccessAdapterChip<F, N> {
    fn set_override_trace_heights(&mut self, overridden_height: usize) {
        self.overridden_height = Some(overridden_height);
    }
    fn add_record(&mut self, record: AccessAdapterRecord<F>) {
        self.records.push(record);
    }
    fn n(&self) -> usize {
        N
    }
    fn generate_trace(self) -> RowMajorMatrix<F>
    where
        F: PrimeField32,
    {
        let mut trace = self.trace;
        let height = trace.height();
        trace.pad_to_height(height.next_power_of_two(), F::ZERO);
        trace
        // TODO(AG): everything related to the calculated trace height
        // needs to be in memory controller, who owns these traces.

        // let width = BaseAir::<F>::width(&self.air);
        // let height = if let Some(oh) = self.overridden_height {
        //     assert!(
        //         oh >= self.records.len(),
        //         "Overridden height is less than the required height"
        //     );
        //     oh
        // } else {
        //     self.records.len()
        // };
        // let height = next_power_of_two_or_zero(height);
        // let mut values = F::zero_vec(height * width);

        // values
        //     .par_chunks_mut(width)
        //     .zip(self.records.into_par_iter())
        //     .for_each(|(row, record)| {
        //         let row: &mut AccessAdapterCols<F, N> = row.borrow_mut();

        //         row.is_valid = F::ONE;
        //         row.values = record.data.try_into().unwrap();
        //         row.address = MemoryAddress::new(record.address_space, record.start_index);

        //         let (left_timestamp, right_timestamp) = match record.kind {
        //             AccessAdapterRecordKind::Split => (record.timestamp, record.timestamp),
        //             AccessAdapterRecordKind::Merge {
        //                 left_timestamp,
        //                 right_timestamp,
        //             } => (left_timestamp, right_timestamp),
        //         };
        //         debug_assert_eq!(max(left_timestamp, right_timestamp), record.timestamp);

        //         row.left_timestamp = F::from_canonical_u32(left_timestamp);
        //         row.right_timestamp = F::from_canonical_u32(right_timestamp);
        //         row.is_split = F::from_bool(record.kind == AccessAdapterRecordKind::Split);

        //         self.air.lt_air.generate_subrow(
        //             (self.range_checker.as_ref(), left_timestamp, right_timestamp),
        //             (&mut row.lt_aux, &mut row.is_right_larger),
        //         );
        //     });
        // RowMajorMatrix::new(values, width)
    }

    fn set_trace(&mut self, trace: RowMajorMatrix<F>) {
        self.trace = trace;
    }

    fn execute_split(
        &mut self,
        address: MemoryAddress<u32, u32>,
        values: &[F],
        timestamp: u32,
        row_slice: &mut [F],
    ) where
        F: PrimeField32,
    {
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
        row_slice: &mut [F],
    ) where
        F: PrimeField32,
    {
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
        self.records.len()
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

pub struct AdapterInventoryTraceCursor<F> {
    // [AG] TODO: replace with a pre-allocated space
    pub cursors: Vec<Cursor<Vec<F>>>,
    pub widths: Vec<usize>,
}

impl<F: PrimeField32> AdapterInventoryTraceCursor<F> {
    pub fn new(as_cnt: usize) -> Self {
        let cursors = vec![Cursor::new(Vec::new()); as_cnt];
        let widths = vec![
            size_of::<AccessAdapterCols<u8, 2>>(),
            size_of::<AccessAdapterCols<u8, 4>>(),
            size_of::<AccessAdapterCols<u8, 8>>(),
            size_of::<AccessAdapterCols<u8, 16>>(),
            size_of::<AccessAdapterCols<u8, 32>>(),
        ];
        Self { cursors, widths }
    }

    pub fn get_row_slice(&mut self, block_size: usize) -> &mut [F] {
        let index = get_chip_index(block_size);
        let begin = self.cursors[index].position() as usize;
        let end = begin + self.widths[index];
        self.cursors[index].get_mut().resize(end, F::ZERO);
        self.cursors[index].set_position(end as u64);
        &mut self.cursors[index].get_mut()[begin..end]
    }

    pub fn extract_trace(&mut self, index: usize) -> Vec<F> {
        std::mem::replace(&mut self.cursors[index], Cursor::new(Vec::new())).into_inner()
    }

    pub fn width(&self, index: usize) -> usize {
        self.widths[index]
    }
}
