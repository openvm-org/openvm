//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.
use std::{
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
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::read_rv32_register;
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_sha_air::ShaAir;
use openvm_stark_backend::{interaction::BusIndex, p3_field::PrimeField32};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod air;
mod columns;
mod config;
mod trace;

pub use air::*;
pub use columns::*;
pub use config::*;
use openvm_circuit::system::memory::{MemoryController, OfflineMemory, RecordId};

#[cfg(test)]
mod tests;

pub struct ShaVmChip<F: PrimeField32, C: ShaChipConfig> {
    pub air: ShaVmAir<C>,
    /// IO and memory data necessary for each opcode call
    pub records: Vec<ShaRecord<F>>,
    pub offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,

    offset: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShaRecord<F> {
    pub from_state: ExecutionState<F>,
    pub dst_read: RecordId,
    pub src_read: RecordId,
    pub len_read: RecordId,
    pub input_records: Vec<Vec<RecordId>>, // type is like Vec<[RecordId; C::NUM_READ_ROWS]>
    pub input_message: Vec<Vec<Vec<u8>>>, // type is like Vec<[[u8; C::SHA256_READ_SIZE]; C::SHA256_NUM_READ_ROWS]>
    pub digest_write: RecordId,
}

impl<F: PrimeField32, C: ShaChipConfig> ShaVmChip<F, C> {
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
            air: ShaVmAir::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bitwise_lookup_chip.bus(),
                address_bits,
                ShaAir::<C>::new(bitwise_lookup_chip.bus(), self_bus_idx),
                Encoder::new(PaddingFlags::COUNT, 2, false),
            ),
            bitwise_lookup_chip,
            records: Vec::new(),
            offset,
            offline_memory,
        }
    }
}

impl<F: PrimeField32, C: ShaChipConfig> InstructionExecutor<F> for ShaVmChip<F, C> {
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

        // need to pad with one 1 bit, 64 bits for the message length and then pad until the length is divisible by [SHA256_BLOCK_BITS]
        let num_blocks = ((len << 3) as usize + 1 + 64).div_ceil(C::BLOCK_BITS);

        // we will read [num_blocks] * [SHA256_BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(src as usize + num_blocks * C::BLOCK_CELLS <= (1 << self.air.ptr_max_bits));
        let mut hasher = Sha256::new();
        let mut input_records = Vec::with_capacity(num_blocks * C::NUM_READ_ROWS);
        let mut input_message = Vec::with_capacity(num_blocks * C::NUM_READ_ROWS);
        let mut read_ptr = src;
        for _ in 0..num_blocks {
            let block_reads_records = (0..C::NUM_READ_ROWS)
                .map(|i| {
                    memory.read::<C::READ_SIZE>(
                        e,
                        F::from_canonical_u32(read_ptr + (i * C::READ_SIZE) as u32),
                    )
                })
                .collect::<Vec<_>>();
            let block_reads_bytes = (0..C::NUM_READ_ROWS)
                .map(|i| {
                    // we add to the hasher only the bytes that are part of the message
                    let num_reads =
                        min(C::READ_SIZE, (max(read_ptr, src + len) - read_ptr) as usize);
                    let row_input: &[u8] = &block_reads_records[i]
                        .1
                        .map(|x| x.as_canonical_u32().try_into().unwrap());
                    hasher.update(&row_input[..num_reads]);
                    read_ptr += C::READ_SIZE as u32;
                    row_input.to_vec()
                })
                .collect::<Vec<_>>();
            input_records.push(block_reads_records.into_iter().map(|x| x.0).collect());
            input_message.push(block_reads_bytes);
        }

        let mut digest = [0u8; SHA_WRITE_SIZE];
        digest.copy_from_slice(hasher.finalize().as_ref());
        let (digest_write, _) = memory.write(
            e,
            F::from_canonical_u32(dst),
            digest.map(|b| F::from_canonical_u8(b)),
        );

        self.records.push(ShaRecord {
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
        C::OPCODE_NAME.to_string()
    }
}

pub fn sha256_solve(input_message: &[u8]) -> [u8; SHA_WRITE_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(input_message);
    let mut output = [0u8; SHA_WRITE_SIZE];
    output.copy_from_slice(hasher.finalize().as_ref());
    output
}
