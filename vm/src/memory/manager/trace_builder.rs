use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;

use super::{operation::MemoryOperation, MemoryAccess, MemoryManager};
use crate::memory::{
    compose, decompose,
    offline_checker::{bridge::MemoryOfflineChecker, columns::MemoryOfflineCheckerAuxCols},
    OpType,
};

pub struct MemoryTraceBuilder<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
    range_checker: Arc<RangeCheckerGateChip>,
    offline_checker: MemoryOfflineChecker,

    accesses_buffer: Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryTraceBuilder<NUM_WORDS, WORD_SIZE, F>
{
    pub fn new(
        memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
        range_checker: Arc<RangeCheckerGateChip>,
        offline_checker: MemoryOfflineChecker,
    ) -> Self {
        Self {
            memory_manager,
            range_checker,
            offline_checker,
            accesses_buffer: Vec::new(),
        }
    }

    pub fn read_word(&mut self, addr_space: F, pointer: F) -> MemoryOperation<WORD_SIZE, F> {
        let mem_access = self
            .memory_manager
            .borrow_mut()
            .read_word(addr_space, pointer);
        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));

        mem_access.op
    }

    pub fn write_word(
        &mut self,
        addr_space: F,
        pointer: F,
        data: [F; WORD_SIZE],
    ) -> MemoryOperation<WORD_SIZE, F> {
        let mem_access = self
            .memory_manager
            .borrow_mut()
            .write_word(addr_space, pointer, data);
        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));

        mem_access.op
    }

    pub fn read_elem(&mut self, addr_space: F, pointer: F) -> F {
        compose(self.read_word(addr_space, pointer).cell.data)
    }

    pub fn write_elem(&mut self, addr_space: F, pointer: F, data: F) {
        self.write_word(addr_space, pointer, decompose(data));
    }

    // pub fn write_elem(&mut self, addr_space: F, pointer: F, data: F) {
    //     self.write_word()
    // }

    pub fn disabled_op(&mut self, addr_space: F, op_type: OpType) -> MemoryOperation<WORD_SIZE, F> {
        let mem_access = self
            .memory_manager
            .borrow_mut()
            .disabled_op(addr_space, op_type);
        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));

        mem_access.op
    }

    pub fn increment_clk(&mut self) {
        self.memory_manager.borrow_mut().increment_clk();
    }

    pub fn take_accesses_buffer(&mut self) -> Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>> {
        std::mem::take(&mut self.accesses_buffer)
    }

    pub fn aux_col_from_access(
        &self,
        access: &MemoryAccess<WORD_SIZE, F>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        Self::memory_access_to_checker_aux_cols(
            &self.offline_checker,
            self.range_checker.clone(),
            access,
        )
    }

    pub fn memory_access_to_checker_aux_cols(
        offline_checker: &MemoryOfflineChecker,
        range_checker: Arc<RangeCheckerGateChip>,
        memory_access: &MemoryAccess<WORD_SIZE, F>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        let timestamp_prev = memory_access.old_cell.clk.as_canonical_u32();
        let timestamp = memory_access.op.cell.clk.as_canonical_u32();

        debug_assert!(timestamp_prev < timestamp);
        let clk_lt_cols = LocalTraceInstructions::generate_trace_row(
            &offline_checker.timestamp_lt_air,
            (timestamp_prev, timestamp, range_checker),
        );

        let addr_space_is_zero_cols = LocalTraceInstructions::generate_trace_row(
            &offline_checker.is_zero_air,
            memory_access.op.addr_space,
        );

        MemoryOfflineCheckerAuxCols::new(
            memory_access.old_cell,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            clk_lt_cols.io.less_than,
            clk_lt_cols.aux,
        )
    }
}
