use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;

use super::{operation::MemoryOperation, MemoryAccess, MemoryManager};
use crate::memory::{
    compose, decompose,
    offline_checker::{bridge::MemoryOfflineChecker, columns::MemoryOfflineCheckerAuxCols},
    OpType,
};

const WORD_SIZE: usize = 1;

// TODO[jpw]: use &'a mut [MemoryOfflineCheckerAuxCols<WORD_SIZE, F>] and allow loading mutable buffers
/// The [MemoryTraceBuilder] uses a buffer to help fill in the auxiliary trace values for memory accesses.
/// Since it uses a buffer, it must be created within a trace generation function and is not intended to be
/// owned by a chip.
#[derive(Clone, Debug)]
pub struct MemoryTraceBuilder<F: PrimeField32> {
    memory_manager: Rc<RefCell<MemoryManager<F>>>,
    // Derived from memory_manager:
    offline_checker: MemoryOfflineChecker,
    range_checker: Arc<RangeCheckerGateChip>,

    accesses_buffer: Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>>,
}

impl<F: PrimeField32> MemoryTraceBuilder<F> {
    pub fn new(memory_manager: Rc<RefCell<MemoryManager<F>>>) -> Self {
        let offline_checker = memory_manager.borrow().make_offline_checker();
        let range_checker = memory_manager.borrow().range_checker.clone();
        Self {
            memory_manager,
            offline_checker,
            range_checker,
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

    // TODO[jpw]: we can default to addr_space = 1 after is_immediate checks are moved out of default memory access
    pub fn disabled_read(&mut self, addr_space: F) -> MemoryOperation<WORD_SIZE, F> {
        self.disabled_op(addr_space, OpType::Read)
    }

    // TODO[jpw]: we can default to addr_space = 1 after is_immediate checks are moved out of default memory access
    pub fn disabled_write(&mut self, addr_space: F) -> MemoryOperation<WORD_SIZE, F> {
        self.disabled_op(addr_space, OpType::Write)
    }

    pub fn disabled_op(&mut self, addr_space: F, op_type: OpType) -> MemoryOperation<WORD_SIZE, F> {
        debug_assert_ne!(
            addr_space,
            F::zero(),
            "Disabled memory operation cannot be immediate"
        );
        let clk = self.memory_manager.borrow().timestamp();
        let mem_access = MemoryAccess::disabled_op(clk, addr_space, op_type);

        self.accesses_buffer
            .push(self.aux_col_from_access(&mem_access));

        mem_access.op
    }

    // TODO[jpw]: rename increment_timestamp
    pub fn increment_clk(&mut self) {
        self.memory_manager.borrow_mut().increment_timestamp();
    }

    pub fn take_accesses_buffer(&mut self) -> Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>> {
        std::mem::take(&mut self.accesses_buffer)
    }

    pub fn aux_col_from_access(
        &self,
        access: &MemoryAccess<WORD_SIZE, F>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        let range_checker = self.range_checker.clone();
        Self::memory_access_to_checker_aux_cols(&self.offline_checker, range_checker, access)
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

        let addr_space_is_zero_cols = offline_checker
            .is_zero_air
            .generate_trace_row(memory_access.op.addr_space);

        MemoryOfflineCheckerAuxCols::new(
            memory_access.old_cell,
            addr_space_is_zero_cols.io.is_zero,
            addr_space_is_zero_cols.inv,
            clk_lt_cols.io.less_than,
            clk_lt_cols.aux,
        )
    }
}
