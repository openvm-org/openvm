//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.
use std::{
    array,
    cmp::{max, min},
    sync::{Arc, Mutex},
};

use openvm_circuit::arch::{
    ExecutionBridge, ExecutionError, ExecutionState, InstructionExecutor, SystemPort,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::read_rv32_register;
use openvm_sha256_air::{Sha256Air, SHA256_BLOCK_BITS};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::{interaction::BusIndex, p3_field::PrimeField32};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod air;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;
use openvm_circuit::system::memory::{MemoryController, OfflineMemory, RecordId};

#[cfg(test)]
mod tests;

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
const SHA256_REGISTER_READS: usize = 3;
/// Number of cells to read in a single memory access
const SHA256_READ_SIZE: usize = 16;
/// Number of cells to write in a single memory access
const SHA256_WRITE_SIZE: usize = 32;
/// Number of rv32 cells read in a SHA256 block
pub const SHA256_BLOCK_CELLS: usize = SHA256_BLOCK_BITS / RV32_CELL_BITS;
/// Number of rows we will do a read on for each SHA256 block
pub const SHA256_NUM_READ_ROWS: usize = SHA256_BLOCK_CELLS / SHA256_READ_SIZE;
pub struct Sha256VmChip<F: PrimeField32> {
    pub air: Sha256VmAir,
    /// IO and memory data necessary for each opcode call
    pub records: Vec<Sha256Record<F>>,
    pub offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,

    offset: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Sha256Record<F> {
    pub from_state: ExecutionState<F>,
    pub dst_read: RecordId,
    pub src_read: RecordId,
    pub len_read: RecordId,
    pub input_records: Vec<[RecordId; SHA256_NUM_READ_ROWS]>,
    pub input_message: Vec<[[u8; SHA256_READ_SIZE]; SHA256_NUM_READ_ROWS]>,
    pub digest_write: RecordId,
}

impl<F: PrimeField32> Sha256VmChip<F> {
    pub fn new(
        SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        }: SystemPort,
        address_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
        self_bus_idx: BusIndex,
        offset: usize,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        Self {
            air: Sha256VmAir::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bitwise_lookup_chip.bus(),
                address_bits,
                Sha256Air::new(bitwise_lookup_chip.bus(), self_bus_idx),
                Encoder::new(PaddingFlags::COUNT, 2, false),
            ),
            bitwise_lookup_chip,
            records: Vec::new(),
            offset,
            offline_memory,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for Sha256VmChip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let local_opcode = opcode.local_opcode_idx(self.offset);
        debug_assert_eq!(local_opcode, Rv32Sha256Opcode::SHA256.local_usize());
        debug_assert_eq!(d, F::from_canonical_u32(RV32_REGISTER_AS));
        debug_assert_eq!(e, F::from_canonical_u32(RV32_MEMORY_AS));

        debug_assert_eq!(from_state.timestamp, memory.timestamp());

        let (dst_read, dst) = read_rv32_register(memory, d, a);
        let (src_read, src) = read_rv32_register(memory, d, b);
        let (len_read, len) = read_rv32_register(memory, d, c);

        #[cfg(debug_assertions)]
        {
            assert!(dst < (1 << self.air.ptr_max_bits));
            assert!(src < (1 << self.air.ptr_max_bits));
            assert!(len < (1 << self.air.ptr_max_bits));
        }

        // need to pad with one 1 bit, 64 bits for the message length and then pad until the length
        // is divisible by [SHA256_BLOCK_BITS]
        let num_blocks = ((len << 3) as usize + 1 + 64).div_ceil(SHA256_BLOCK_BITS);

        // we will read [num_blocks] * [SHA256_BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(
            src as usize + num_blocks * SHA256_BLOCK_CELLS <= (1 << self.air.ptr_max_bits)
        );
        let mut hasher = Sha256::new();
        let mut input_records = Vec::with_capacity(num_blocks * SHA256_NUM_READ_ROWS);
        let mut input_message = Vec::with_capacity(num_blocks * SHA256_NUM_READ_ROWS);
        let mut read_ptr = src;
        for _ in 0..num_blocks {
            let block_reads_records = array::from_fn(|i| {
                memory.read::<u8, SHA256_READ_SIZE>(
                    e,
                    F::from_canonical_u32(read_ptr + (i * SHA256_READ_SIZE) as u32),
                )
            });
            let block_reads_bytes = array::from_fn(|i| {
                // we add to the hasher only the bytes that are part of the message
                let num_reads = min(
                    SHA256_READ_SIZE,
                    (max(read_ptr, src + len) - read_ptr) as usize,
                );
                let row_input = block_reads_records[i].1;
                hasher.update(&row_input[..num_reads]);
                read_ptr += SHA256_READ_SIZE as u32;
                row_input
            });
            input_records.push(block_reads_records.map(|x| x.0));
            input_message.push(block_reads_bytes);
        }

        let mut digest = [0u8; SHA256_WRITE_SIZE];
        digest.copy_from_slice(hasher.finalize().as_ref());
        let (digest_write, _) = memory.write(e, F::from_canonical_u32(dst), &digest);

        self.records.push(Sha256Record {
            from_state: from_state.map(F::from_canonical_u32),
            dst_read,
            src_read,
            len_read,
            input_records,
            input_message,
            digest_write,
        });

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory.timestamp(),
        })
    }

    fn get_opcode_name(&self, _: usize) -> String {
        "SHA256".to_string()
    }
}

pub fn sha256_solve(input_message: &[u8]) -> [u8; SHA256_WRITE_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(input_message);
    let mut output = [0u8; SHA256_WRITE_SIZE];
    output.copy_from_slice(hasher.finalize().as_ref());
    output
}
