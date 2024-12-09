use std::{array::from_fn, borrow::BorrowMut as _, cell::RefCell, mem::size_of, sync::Arc};

use air::{DummyMemoryInteractionCols, MemoryDummyAir};
use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionType,
    p3_field::{AbstractField, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    prover::types::AirProofInput,
    rap::AnyRap,
    Chip, ChipUsageGetter,
};
use rand::{seq::SliceRandom, Rng};

use crate::system::memory::{
    offline_checker::{MemoryBus, MemoryBusInteraction},
    MemoryAddress, MemoryControllerRef,
};

pub mod air;

const WORD_SIZE: usize = 1;

/// A dummy testing chip that will add unconstrained messages into the [MemoryBus].
/// Stores a log of raw messages to send/receive to the [MemoryBus].
///
/// It will create a [air::MemoryDummyAir] to add messages to MemoryBus.
#[derive(Debug)]
pub struct MemoryTester<F> {
    pub bus: MemoryBus,
    pub controller: MemoryControllerRef<F>,
    /// Log of raw bus messages
    pub records: Vec<MemoryBusInteraction<F>>,
}

impl<F: PrimeField32> MemoryTester<F> {
    pub fn new(controller: MemoryControllerRef<F>) -> Self {
        let bus = controller.borrow().memory_bus;
        Self {
            bus,
            controller,
            records: Vec::new(),
        }
    }

    /// Returns the cell value at the current timestamp according to [MemoryController].
    pub fn read_cell(&mut self, address_space: usize, pointer: usize) -> F {
        let [addr_space, pointer] = [address_space, pointer].map(F::from_canonical_usize);
        // core::BorrowMut confuses compiler
        let read = RefCell::borrow_mut(&self.controller).read_cell(addr_space, pointer);
        let address = MemoryAddress::new(addr_space, pointer);
        self.records.push(self.bus.receive(
            address,
            read.data.to_vec(),
            F::from_canonical_u32(read.prev_timestamp),
        ));
        self.records.push(self.bus.send(
            address,
            read.data.to_vec(),
            F::from_canonical_u32(read.timestamp),
        ));
        read.value()
    }

    pub fn write_cell(&mut self, address_space: usize, pointer: usize, value: F) {
        let [addr_space, pointer] = [address_space, pointer].map(F::from_canonical_usize);
        let write = RefCell::borrow_mut(&self.controller).write_cell(addr_space, pointer, value);
        let address = MemoryAddress::new(addr_space, pointer);
        self.records.push(self.bus.receive(
            address,
            write.prev_data.to_vec(),
            F::from_canonical_u32(write.prev_timestamp),
        ));
        self.records.push(self.bus.send(
            address,
            write.data.to_vec(),
            F::from_canonical_u32(write.timestamp),
        ));
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

impl<SC: StarkGenericConfig> Chip<SC> for MemoryTester<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(MemoryDummyAir::<WORD_SIZE>::new(self.bus))
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let height = self.records.len().next_power_of_two();
        let width = self.trace_width();
        let mut values = Val::<SC>::zero_vec(height * width);
        // This zip only goes through records. The padding rows between records.len()..height
        // are filled with zeros - in particular count = 0 so nothing is added to bus.
        for (row, record) in values.chunks_mut(width).zip(self.records) {
            let row: &mut DummyMemoryInteractionCols<Val<SC>, WORD_SIZE> = row.borrow_mut();
            row.address = record.address;
            row.data = record.data.try_into().unwrap();
            row.timestamp = record.timestamp;
            row.count = match record.interaction_type {
                InteractionType::Send => Val::<SC>::ONE,
                InteractionType::Receive => -Val::<SC>::ONE,
            };
        }
        AirProofInput::simple_no_pis(air, RowMajorMatrix::new(values, width))
    }
}

impl<F: PrimeField32> ChipUsageGetter for MemoryTester<F> {
    fn air_name(&self) -> String {
        "MemoryDummyAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        self.records.len() / self.trace_width()
    }

    fn trace_width(&self) -> usize {
        size_of::<DummyMemoryInteractionCols<u8, WORD_SIZE>>()
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
    rng.gen_range(0..MAX_MEMORY - len) / len * len
}
