use p3_field::PrimeField32;

use super::{MemoryAccess, MemoryChipRef, MemoryReadRecord, MemoryWriteRecord};
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
        let read = self.memory_chip.borrow_mut().read(addr_space, pointer);

        let mem_access = MemoryAccess::from_read(read.clone());

        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));

        read
    }

    pub fn write_cell(
        &mut self,
        addr_space: F,
        pointer: F,
        data: F,
    ) -> MemoryWriteRecord<WORD_SIZE, F> {
        let write = self
            .memory_chip
            .borrow_mut()
            .write(addr_space, pointer, data);

        let mem_access = MemoryAccess::from_write(write.clone());

        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));

        write
    }

    pub fn read_elem(&mut self, addr_space: F, pointer: F) -> F {
        self.read_cell(addr_space, pointer).data[0]
    }

    pub fn disabled_op(&mut self, addr_space: F) {
        debug_assert_ne!(
            addr_space,
            F::zero(),
            "Disabled memory operation cannot be immediate"
        );
        let clk = self.memory_chip.borrow().timestamp();
        let mem_access = MemoryAccess::disabled_op(clk, addr_space);

        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));
    }

    // TODO[jpw]: rename increment_timestamp
    pub fn increment_clk(&mut self) {
        self.memory_chip.borrow_mut().increment_timestamp();
    }

    pub fn take_accesses_buffer(&mut self) -> Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>> {
        std::mem::take(&mut self.accesses_buffer)
    }

    pub fn aux_col_from_access(
        &self,
        access: &MemoryAccess<WORD_SIZE, F>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        self.memory_chip.borrow().make_access_cols(access.clone())
    }
}
