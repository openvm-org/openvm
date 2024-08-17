use std::sync::Arc;

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use p3_field::PrimeField32;
use parking_lot::Mutex;

use super::{operation::MemoryOperation, MemoryManager};
use crate::memory::{
    compose, decompose,
    offline_checker::{
        air::NewMemoryOfflineChecker,
        columns::{MemoryOfflineCheckerAuxCols, NewMemoryAccess},
    },
    OpType,
};

pub struct MemoryTraceBuilder<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    memory_manager: Arc<Mutex<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
    range_checker: Arc<RangeCheckerGateChip>,
    offline_checker: NewMemoryOfflineChecker<WORD_SIZE>,

    accesses_buffer: Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryTraceBuilder<NUM_WORDS, WORD_SIZE, F>
{
    pub fn new(
        memory_manager: Arc<Mutex<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
        range_checker: Arc<RangeCheckerGateChip>,
        offline_checker: NewMemoryOfflineChecker<WORD_SIZE>,
    ) -> Self {
        Self {
            memory_manager,
            range_checker,
            offline_checker,
            accesses_buffer: Vec::new(),
        }
    }

    pub fn read_word(&mut self, addr_space: F, pointer: F) -> MemoryOperation<WORD_SIZE, F> {
        let mem_access = self.memory_manager.lock().read_word(addr_space, pointer);
        self.accesses_buffer
            .push(self.memory_access_to_checker_aux_cols(&mem_access));

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
            .lock()
            .write_word(addr_space, pointer, data);
        self.accesses_buffer
            .push(self.memory_access_to_checker_aux_cols(&mem_access));

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
        let mem_access = self.memory_manager.lock().disabled_op(addr_space, op_type);
        self.accesses_buffer
            .push(self.memory_access_to_checker_aux_cols(&mem_access));

        mem_access.op
    }

    pub fn take_accesses_buffer(&mut self) -> Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, F>> {
        std::mem::take(&mut self.accesses_buffer)
    }

    fn memory_access_to_checker_aux_cols(
        &self,
        memory_access: &NewMemoryAccess<WORD_SIZE, F>,
    ) -> MemoryOfflineCheckerAuxCols<WORD_SIZE, F> {
        let clk_lt_cols = LocalTraceInstructions::generate_trace_row(
            &self.offline_checker.clk_lt_air,
            (
                memory_access.old_cell.clk.as_canonical_u32(),
                memory_access.op.cell.clk.as_canonical_u32(),
                self.range_checker.clone(),
            ),
        );

        let addr_space_is_zero_cols = LocalTraceInstructions::generate_trace_row(
            &self.offline_checker.is_zero_air,
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
