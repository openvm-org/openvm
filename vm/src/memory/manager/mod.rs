use std::{collections::HashMap, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use self::{access_cell::AccessCell, interface::MemoryInterface};
use super::{audit::MemoryAuditChip, offline_checker::columns::NewMemoryAccess};
use crate::{
    memory::{decompose, manager::operation::MemoryOperation, OpType},
    vm::config::MemoryConfig,
};

pub mod access;
pub mod access_cell;
pub mod dimensions;
pub mod interface;
pub mod operation;
pub mod trace_builder;

pub struct MemoryManager<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    pub interface_chip: MemoryInterface<NUM_WORDS, WORD_SIZE, F>,
    clk: F,
    /// Maps (addr_space, pointer) to (data, timestamp)
    memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryManager<NUM_WORDS, WORD_SIZE, F>
{
    // TODO[osama]: move to the bottom
    pub fn get_clk(&self) -> F {
        self.clk
    }

    // pub fn with_persistent_memory(
    //     memory_dimensions: MemoryDimensions,
    //     memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
    // ) -> Self {
    //     Self {
    //         interface_chip: MemoryInterface::Persistent(MemoryExpandInterfaceChip::new(
    //             memory_dimensions,
    //         )),
    //         clk: F::one(),
    //         memory,
    //     }
    // }

    pub fn with_volatile_memory(
        mem_config: MemoryConfig,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            interface_chip: MemoryInterface::Volatile(MemoryAuditChip::new(
                mem_config.addr_space_max_bits,
                mem_config.pointer_max_bits,
                mem_config.decomp,
                range_checker,
            )),
            clk: F::one(),
            memory: HashMap::new(),
        }
    }

    pub fn read_word(&mut self, addr_space: F, pointer: F) -> NewMemoryAccess<WORD_SIZE, F> {
        let cur_clk = self.clk;
        self.clk += F::one();

        if addr_space == F::zero() {
            let data = decompose(pointer);
            return NewMemoryAccess::<WORD_SIZE, F>::new(
                MemoryOperation::new(
                    addr_space,
                    pointer,
                    F::from_canonical_u8(OpType::Read as u8),
                    AccessCell::new(data, cur_clk),
                    F::one(),
                ),
                AccessCell::new(data, cur_clk),
            );
        }
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let cell = self.memory.get_mut(&(addr_space, pointer)).unwrap();
        let (old_clk, old_data) = (cell.clk, cell.data);
        // TODO[osama]: remove
        assert!(old_clk < cur_clk);

        // Updating AccessCell
        cell.clk = cur_clk;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data, cur_clk);

        NewMemoryAccess::<WORD_SIZE, F>::new(
            MemoryOperation::new(
                addr_space,
                pointer,
                F::from_canonical_u8(OpType::Read as u8),
                *cell,
                F::one(),
            ),
            AccessCell::new(old_data, old_clk),
        )
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_word(&self, address_space: F, pointer: F) -> [F; WORD_SIZE] {
        self.memory.get(&(address_space, pointer)).unwrap().data
    }

    pub fn write_word(
        &mut self,
        addr_space: F,
        pointer: F,
        data: [F; WORD_SIZE],
    ) -> NewMemoryAccess<WORD_SIZE, F> {
        assert!(addr_space != F::zero());
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let cur_clk = self.clk;
        self.clk += F::one();

        let cell = self
            .memory
            .entry((addr_space, pointer))
            .or_insert(AccessCell {
                data: [F::zero(); WORD_SIZE],
                clk: F::zero(),
            });
        let (old_clk, old_data) = (cell.clk, cell.data);
        if old_clk >= cur_clk {
            println!("interesting");
            println!("old_clk: {:?}, clk: {:?}", old_clk, cur_clk);
        }
        assert!(old_clk < cur_clk);

        // Updating AccessCell
        cell.clk = cur_clk;
        cell.data = data;

        self.interface_chip
            .touch_address(addr_space, pointer, old_data, old_clk);

        NewMemoryAccess::<WORD_SIZE, F>::new(
            MemoryOperation::new(
                addr_space,
                pointer,
                F::from_canonical_u8(OpType::Write as u8),
                *cell,
                F::one(),
            ),
            AccessCell::new(old_data, old_clk),
        )
    }

    pub fn unsafe_write_word(&mut self, addr_space: F, pointer: F, data: [F; WORD_SIZE]) {
        assert!(addr_space != F::zero());
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        self.memory
            .entry((addr_space, pointer))
            .and_modify(|cell| cell.data = data)
            .or_insert(AccessCell {
                data,
                clk: F::zero(),
            });
    }

    pub fn generate_memory_interface_trace(&self) -> RowMajorMatrix<F> {
        let all_addresses = self.interface_chip.all_addresses();
        let mut final_memory = HashMap::new();
        for (addr_space, pointer) in all_addresses {
            final_memory.insert(
                (addr_space, pointer),
                *self.memory.get(&(addr_space, pointer)).unwrap(),
            );
        }

        self.interface_chip
            .generate_trace(final_memory, (2 * self.memory.len()).next_power_of_two())
    }

    pub fn disabled_op(&mut self, addr_space: F, op_type: OpType) -> NewMemoryAccess<WORD_SIZE, F> {
        let cur_clk = self.clk;
        self.clk += F::one();

        NewMemoryAccess::<WORD_SIZE, F>::new(
            MemoryOperation::new(
                addr_space,
                F::zero(),
                F::from_canonical_u8(op_type as u8),
                AccessCell::new([F::zero(); WORD_SIZE], cur_clk),
                F::zero(),
            ),
            AccessCell::new([F::zero(); WORD_SIZE], F::zero()),
        )
    }

    // pub fn memory_clone(&self) -> HashMap<(F, F), F> {
    //     self.memory.clone()
    // }

    // pub fn read_elem(&mut self, timestamp: usize, address_space: F, address: F) -> F {
    //     compose(self.read_word(timestamp, address_space, address))
    // }

    // /// Reads an element directly from memory without updating internal state.
    // ///
    // /// Any value returned is unconstrained.
    // pub fn unsafe_read_elem(&self, address_space: F, address: F) -> F {
    //     compose(self.unsafe_read_word(address_space, address))
    // }

    // pub fn write_elem(&mut self, timestamp: usize, address_space: F, address: F, data: F) {
    //     self.write_word(timestamp, address_space, address, decompose(data));
    // }
}
