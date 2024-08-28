use std::{array::from_fn, borrow::BorrowMut as _, cell::RefCell, mem::size_of};

use afs_stark_backend::rap::AnyRap;
use air::{DummyMemoryInteractionCols, MemoryDummyAir};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use rand::{seq::SliceRandom, Rng};

use crate::{
    arch::chips::MachineChip,
    memory::{
        manager::MemoryChipRef,
        offline_checker::bus::{MemoryBus, MemoryBusInteraction},
        MemoryAddress, OpType,
    },
};

pub mod air;

const WORD_SIZE: usize = 1;

/// A dummy testing chip that will add unconstrained messages into the [MemoryBus].
/// Stores a log of raw messages to send/receive to the [MemoryBus].
///
/// It will create a [air::MemoryDummyAir] to add messages to MemoryBus.
#[derive(Clone, Debug)]
pub struct MemoryTester<F: PrimeField32> {
    pub bus: MemoryBus,
    pub chip: MemoryChipRef<F>,
    /// Log of raw bus messages
    pub records: Vec<MemoryBusInteraction<F, 1>>,
}

impl<F: PrimeField32> MemoryTester<F> {
    pub fn new(chip: MemoryChipRef<F>) -> Self {
        let bus = chip.borrow().memory_bus;
        Self {
            bus,
            chip,
            records: Vec::new(),
        }
    }

    /// Returns the cell value at the current timestamp according to [MemoryChip].
    pub fn read_cell(&mut self, address_space: usize, pointer: usize) -> F {
        let [addr_space, pointer] = [address_space, pointer].map(F::from_canonical_usize);
        // core::BorrowMut confuses compiler
        let read = RefCell::borrow_mut(&self.chip).read(addr_space, pointer);
        let address = MemoryAddress::new(addr_space, pointer);
        self.records
            .push(self.bus.read(address, read.data, read.prev_timestamp));
        self.records
            .push(self.bus.write(address, read.data, read.timestamp));
        read.value()
    }

    pub fn write_cell(&mut self, address_space: usize, pointer: usize, value: F) {
        let [addr_space, pointer] = [address_space, pointer].map(F::from_canonical_usize);
        let write = RefCell::borrow_mut(&self.chip).write(addr_space, pointer, value);
        let address = MemoryAddress::new(addr_space, pointer);
        self.records.push(
            self.bus
                .read(address, write.prev_data, write.prev_timestamp),
        );
        self.records
            .push(self.bus.write(address, write.data, write.timestamp));
    }

    pub fn read<const N: usize>(&mut self, address_space: usize, pointer: usize) -> [F; N] {
        from_fn(|i| self.read_cell(address_space, pointer + i))
    }

    pub fn write<const N: usize>(
        &mut self,
        address_space: usize,
        mut pointer: usize,
        cells: [F; N],
    ) {
        for cell in cells {
            self.write_cell(address_space, pointer, cell);
            pointer += 1;
        }
    }
}

impl<F: PrimeField32> MachineChip<F> for MemoryTester<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let height = self.records.len().next_power_of_two();
        let width = self.trace_width();
        let mut values = vec![F::zero(); height * width];
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros - in particular count = 0 so nothing is added to bus.
        for (row, record) in values.chunks_mut(width).zip(&self.records) {
            let row: &mut DummyMemoryInteractionCols<F, WORD_SIZE> = row.borrow_mut();
            row.address = record.address;
            row.data = record.data;
            row.timestamp = record.timestamp;
            row.count = if record.op_type == OpType::Write {
                F::one()
            } else {
                -F::one()
            };
        }
        RowMajorMatrix::new(values, self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(MemoryDummyAir::<WORD_SIZE>::new(self.bus))
    }

    fn current_trace_height(&self) -> usize {
        self.current_trace_cells() / self.trace_width()
    }

    fn trace_width(&self) -> usize {
        size_of::<DummyMemoryInteractionCols<u8, WORD_SIZE>>()
    }

    fn current_trace_cells(&self) -> usize {
        self.records.len()
    }
}

pub fn gen_address_space<R>(rng: &mut R) -> usize
where
    R: Rng + ?Sized,
{
    *[1, 2].choose(rng).unwrap()
}

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.gen_range(0..MAX_MEMORY - len)
}
