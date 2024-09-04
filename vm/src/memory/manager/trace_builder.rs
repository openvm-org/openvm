use p3_field::PrimeField32;

use super::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord};
use crate::memory::offline_checker::columns::MemoryOfflineCheckerAuxCols;

const WORD_SIZE: usize = 1;

// TODO[jpw]: use &'a mut [MemoryOfflineCheckerAuxCols<WORD_SIZE, F>] and allow loading mutable buffers
/// The [MemoryTraceBuilder] uses a buffer to help fill in the auxiliary trace values for memory accesses.
/// Since it uses a buffer, it must be created within a trace generation function and is not intended to be
/// owned by a chip.
#[derive(Clone, Debug)]
pub struct MemoryTraceBuilder<F: PrimeField32> {
    memory_chip: MemoryChipRef<F>,
    accesses_buffer: Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>>,
}

impl<F: PrimeField32> MemoryTraceBuilder<F> {
    pub fn new(memory_chip: MemoryChipRef<F>) -> Self {
        Self {
            memory_chip,
            accesses_buffer: Vec::new(),
        }
    }

    pub fn read_cell(&mut self, addr_space: F, pointer: F) -> MemoryReadRecord<WORD_SIZE, F> {
        let mut memory_chip = self.memory_chip.borrow_mut();
        let read = memory_chip.read_cell(addr_space, pointer);
        let aux_cols = memory_chip.make_read_aux_cols(read.clone());

        self.accesses_buffer.push(aux_cols);

        read
    }

    pub fn write_cell(
        &mut self,
        addr_space: F,
        pointer: F,
        data: F,
    ) -> MemoryWriteRecord<WORD_SIZE, F> {
        let mut memory_chip = self.memory_chip.borrow_mut();
        let write = memory_chip.write_cell(addr_space, pointer, data);
        let aux_cols = memory_chip.make_write_aux_cols(write.clone());

        self.accesses_buffer.push(aux_cols);

        write
    }

    pub fn read_elem(&mut self, addr_space: F, pointer: F) -> F {
        self.read_cell(addr_space, pointer).data[0]
    }

    pub fn disabled_op(&mut self) {
        self.accesses_buffer
            .push(self.memory_chip.borrow().make_disabled_write_aux_cols());
    }

    // TODO[jpw]: rename increment_timestamp
    pub fn increment_clk(&mut self) {
        self.memory_chip.borrow_mut().increment_timestamp();
    }

    pub fn take_accesses_buffer(mut self) -> Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>> {
        std::mem::take(&mut self.accesses_buffer)
    }
}
