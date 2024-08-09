use std::{
    array::from_fn,
    collections::{BTreeMap, HashMap},
};

use afs_primitives::{is_less_than_tuple::IsLessThanTupleAir, offline_checker::OfflineChecker};
use p3_field::{Field, PrimeField32};

use super::{
    audit::air::AuditAir, expand::MemoryDimensions, interface::MemoryInterfaceChip,
    manager::MemoryReadWriteOpCols, MemoryAccess,
};
use crate::{
    cpu::{MEMORY_BUS, RANGE_CHECKER_BUS},
    memory::{compose, decompose, OpType},
};

mod air;
mod columns;
mod trace;

// TODO[osama]: move this to another file
pub struct NewMemoryChip<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    pub interface_chip: MemoryInterfaceChip<NUM_WORDS, WORD_SIZE, F>,
    memory: HashMap<(F, F), F>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    NewMemoryChip<NUM_WORDS, WORD_SIZE, F>
{
    pub fn new(memory_dimensions: MemoryDimensions, memory: HashMap<(F, F), F>) -> Self {
        Self {
            interface_chip: MemoryInterfaceChip::new(memory_dimensions),
            memory,
        }
    }

    pub fn read_word(
        &mut self,
        // TODO[osama]: can this be a field element?
        timestamp: usize,
        addr_space: F,
        pointer: F,
    ) -> MemoryReadWriteOpCols<WORD_SIZE, F> {
        if addr_space == F::zero() {
            let data = decompose(pointer);
            return MemoryReadWriteOpCols::<WORD_SIZE, F>::new(
                addr_space,
                pointer,
                data,
                F::from_canonical_usize(timestamp),
                data,
                F::from_canonical_usize(timestamp),
            );
        }

        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let data = from_fn(|i| self.memory[&(addr_space, pointer + F::from_canonical_usize(i))]);

        self.interface_chip.touch_address(
            addr_space,
            pointer,
            data,
            F::from_canonical_usize(timestamp),
        );

        MemoryReadWriteOpCols::<WORD_SIZE, F>::new(
            addr_space,
            pointer,
            data,
            F::from_canonical_usize(timestamp),
            data,
            F::from_canonical_usize(timestamp + 1),
        )
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_word(&self, address_space: F, pointer: F) -> [F; WORD_SIZE] {
        from_fn(|i| self.memory[&(address_space, pointer + F::from_canonical_usize(i))])
    }

    pub fn write_word(
        &mut self,
        timestamp: usize,
        addr_space: F,
        pointer: F,
        data: [F; WORD_SIZE],
    ) -> MemoryReadWriteOpCols<WORD_SIZE, F> {
        assert!(addr_space != F::zero());
        debug_assert!((pointer.as_canonical_u32() as usize) % WORD_SIZE == 0);

        let old_data =
            from_fn(|i| self.memory[&(addr_space, pointer + F::from_canonical_usize(i))]);
        self.interface_chip.touch_address(
            addr_space,
            pointer,
            old_data,
            F::from_canonical_usize(timestamp),
        );

        for (i, &datum) in data.iter().enumerate() {
            self.memory
                .insert((addr_space, pointer + F::from_canonical_usize(i)), datum);
        }

        MemoryReadWriteOpCols::<WORD_SIZE, F>::new(
            addr_space,
            pointer,
            data,
            F::from_canonical_usize(timestamp),
            data,
            F::from_canonical_usize(timestamp + 1),
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

// ------

pub struct MemoryOfflineChecker {
    pub offline_checker: OfflineChecker,
}

impl MemoryOfflineChecker {
    pub fn air_width(&self) -> usize {
        OfflineChecker::air_width(&self.offline_checker)
    }
}

pub struct MemoryChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub air: MemoryOfflineChecker,
    pub accesses: Vec<MemoryAccess<WORD_SIZE, F>>,
    memory: HashMap<(F, F), F>,
    last_timestamp: Option<usize>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> MemoryChip<WORD_SIZE, F> {
    pub fn new(
        addr_space_limb_bits: usize,
        pointer_limb_bits: usize,
        clk_limb_bits: usize,
        decomp: usize,
        memory: HashMap<(F, F), F>,
    ) -> Self {
        let idx_clk_limb_bits = vec![addr_space_limb_bits, pointer_limb_bits, clk_limb_bits];

        let offline_checker = OfflineChecker::new(
            idx_clk_limb_bits,
            decomp,
            2,
            WORD_SIZE,
            RANGE_CHECKER_BUS,
            MEMORY_BUS,
        );

        Self {
            air: MemoryOfflineChecker { offline_checker },
            accesses: vec![],
            memory,
            last_timestamp: None,
        }
    }

    pub fn read_word(&mut self, timestamp: usize, address_space: F, address: F) -> [F; WORD_SIZE] {
        if address_space == F::zero() {
            return decompose(address);
        }
        if let Some(last_timestamp) = self.last_timestamp {
            assert!(timestamp > last_timestamp);
        }
        self.last_timestamp = Some(timestamp);
        let data = from_fn(|i| self.memory[&(address_space, address + F::from_canonical_usize(i))]);
        self.accesses.push(MemoryAccess {
            timestamp,
            op_type: OpType::Read,
            address_space,
            address,
            data,
        });
        data
    }

    /// Reads a word directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_word(&self, address_space: F, address: F) -> [F; WORD_SIZE] {
        from_fn(|i| self.memory[&(address_space, address + F::from_canonical_usize(i))])
    }

    pub fn write_word(
        &mut self,
        timestamp: usize,
        address_space: F,
        address: F,
        data: [F; WORD_SIZE],
    ) {
        assert!(address_space != F::zero());
        if let Some(last_timestamp) = self.last_timestamp {
            assert!(timestamp > last_timestamp);
        }
        self.last_timestamp = Some(timestamp);
        for (i, &datum) in data.iter().enumerate() {
            self.memory
                .insert((address_space, address + F::from_canonical_usize(i)), datum);
        }
        self.accesses.push(MemoryAccess {
            timestamp,
            op_type: OpType::Write,
            address_space,
            address,
            data,
        });
    }

    pub fn memory_clone(&self) -> HashMap<(F, F), F> {
        self.memory.clone()
    }

    pub fn read_elem(&mut self, timestamp: usize, address_space: F, address: F) -> F {
        compose(self.read_word(timestamp, address_space, address))
    }

    /// Reads an element directly from memory without updating internal state.
    ///
    /// Any value returned is unconstrained.
    pub fn unsafe_read_elem(&self, address_space: F, address: F) -> F {
        compose(self.unsafe_read_word(address_space, address))
    }

    pub fn write_elem(&mut self, timestamp: usize, address_space: F, address: F, data: F) {
        self.write_word(timestamp, address_space, address, decompose(data));
    }

    pub fn current_height(&self) -> usize {
        self.accesses.len()
    }
}
