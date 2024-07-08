use std::{
    array::from_fn,
    collections::{HashMap, HashSet},
};

use afs_chips::is_less_than_tuple::columns::IsLessThanTupleAuxCols;
use p3_field::PrimeField32;

use crate::memory::{compose, decompose, OpType};

use super::MemoryAccess;

mod air;
mod bridge;
mod columns;
mod trace;

pub struct OfflineChecker<const WORD_SIZE: usize> {
    addr_clk_limb_bits: Vec<usize>,
    decomp: usize,
}

impl<const WORD_SIZE: usize> OfflineChecker<WORD_SIZE> {
    pub fn mem_width(&self) -> usize {
        // 1 for addr_space, 1 for pointer, data_len for data
        2 + WORD_SIZE
    }

    pub fn air_width(&self) -> usize {
        10 + self.mem_width()
            + 2 * WORD_SIZE
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.addr_clk_limb_bits.clone(),
                self.decomp,
                3,
            )
    }
}

pub struct MemoryChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub air: OfflineChecker<WORD_SIZE>,
    pub accesses: Vec<MemoryAccess<WORD_SIZE, F>>,
    access_indices: HashSet<(F, F)>,
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
        Self {
            air: OfflineChecker {
                addr_clk_limb_bits: vec![addr_space_limb_bits, pointer_limb_bits, clk_limb_bits],
                decomp,
            },
            accesses: vec![],
            access_indices: HashSet::new(),
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
        self.access_indices.insert((address_space, address));
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
        self.access_indices.insert((address_space, address));
    }

    pub fn write_hint(&mut self, timestamp: usize, address_space: F, address: F, hint: Vec<F>) {
        assert!(address_space != F::zero());
        if let Some(last_timestamp) = self.last_timestamp {
            assert!(timestamp > last_timestamp);
        }
        self.last_timestamp = Some(timestamp);

        for (i, &datum) in hint.iter().enumerate() {
            assert!(!self.access_indices.contains(&(
                address_space,
                address + F::from_canonical_usize(i * WORD_SIZE)
            )));
            let decomp: [F; WORD_SIZE] = decompose(datum);
            for (j, &decomp_elem) in decomp.iter().enumerate() {
                self.memory.insert(
                    (
                        address_space,
                        address + F::from_canonical_usize(i * WORD_SIZE + j),
                    ),
                    decomp_elem,
                );
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
