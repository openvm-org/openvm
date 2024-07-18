use std::collections::hash_map::Entry;
use std::{array::from_fn, collections::HashMap};

use afs_chips::offline_checker::{OfflineChecker, OfflineCheckerChip};
use p3_field::PrimeField32;

use crate::cpu::{MEMORY_BUS, RANGE_CHECKER_BUS};
use crate::memory::{compose, decompose, OpType};

use super::MemoryAccess;

mod trace;

pub struct MemoryChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub chip: OfflineCheckerChip<WORD_SIZE, F, MemoryAccess<WORD_SIZE, F>>,
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
    ) -> Self {
        let idx_clk_limb_bits = vec![addr_space_limb_bits, pointer_limb_bits, clk_limb_bits];

        let offline_checker_air = OfflineChecker::<WORD_SIZE>::new(
            idx_clk_limb_bits,
            decomp,
            2,
            WORD_SIZE,
            RANGE_CHECKER_BUS,
            MEMORY_BUS,
        );

        let offline_checker_chip =
            OfflineCheckerChip::<WORD_SIZE, F, MemoryAccess<WORD_SIZE, F>>::new(
                offline_checker_air,
            );

        Self {
            chip: offline_checker_chip,
            accesses: vec![],
            memory: HashMap::new(),
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

    /// Writes the length and contents of `hint` into memory starting at `e[d[a]]`.
    ///
    /// First writes `hint.len()` into `e[d[a]]`. Then writes `hint` into `e[d[a] + 1, ..., d[a] + hint.len()]`.
    /// Panics if any of these writes are not the first write to the corresponding memory locations.
    pub fn write_hint(&mut self, op_a: F, d: F, e: F, hint: &[F]) {
        let address = if d != F::zero() {
            self.memory[&(d, op_a)]
        } else {
            op_a
        };
        self.init_memory(e, address, F::from_canonical_usize(hint.len()));

        for (i, &datum) in hint.iter().enumerate() {
            self.init_memory(
                e,
                address + F::from_canonical_usize((i + 1) * WORD_SIZE),
                datum,
            );
        }
    }

    fn init_memory(&mut self, addr_space: F, addr: F, value: F) {
        let decomp = decompose::<WORD_SIZE, _>(value);
        for (j, &value) in decomp.iter().enumerate() {
            let loc = (addr_space, addr + F::from_canonical_usize(j));
            match self.memory.entry(loc) {
                Entry::Occupied(_) => panic!(
                    "cannot initialize previously used memory ({}, {})",
                    addr_space, addr
                ),
                Entry::Vacant(v) => {
                    v.insert(value);
                }
            }
        }
    }

    pub fn read_elem(&mut self, timestamp: usize, address_space: F, address: F) -> F {
        compose(self.read_word(timestamp, address_space, address))
    }

    pub fn write_elem(&mut self, timestamp: usize, address_space: F, address: F, data: F) {
        self.write_word(timestamp, address_space, address, decompose(data));
    }
}
