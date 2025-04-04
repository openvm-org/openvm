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
use openvm_sha256_transpiler::Rv32Sha2Opcode;
use openvm_sha_air::{Sha256Config, Sha2Air, Sha512Config};
use openvm_stark_backend::{interaction::BusIndex, p3_field::PrimeField32};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};

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

pub struct Sha2VmChip<F: PrimeField32, C: ShaChipConfig> {
    pub air: Sha2VmAir<C>,
    /// IO and memory data necessary for each opcode call
    pub records: Vec<Sha2Record<F>>,
    pub offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,

    offset: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Sha2Record<F> {
    pub from_state: ExecutionState<F>,
    pub dst_read: RecordId,
    pub src_read: RecordId,
    pub len_read: RecordId,
    pub input_records: Vec<Vec<RecordId>>, // type is like Vec<[RecordId; C::NUM_READ_ROWS]>
    pub input_message: Vec<Vec<Vec<u8>>>, // type is like Vec<[[u8; C::SHA256_READ_SIZE]; C::SHA256_NUM_READ_ROWS]>
    pub digest_writes: Vec<RecordId>,     // one write for sha256, two for sha512
}

impl<F: PrimeField32, C: ShaChipConfig> Sha2VmChip<F, C> {
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
            air: Sha2VmAir::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bitwise_lookup_chip.bus(),
                address_bits,
                Sha2Air::<C>::new(bitwise_lookup_chip.bus(), self_bus_idx),
                Encoder::new(PaddingFlags::COUNT, 2, false),
            ),
            bitwise_lookup_chip,
            records: Vec::new(),
            offset,
            offline_memory,
        }
    }
}

impl<F: PrimeField32, C: ShaChipConfig> InstructionExecutor<F> for Sha2VmChip<F, C> {
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
        debug_assert!(
            local_opcode == Rv32Sha2Opcode::SHA256.local_usize()
                || local_opcode == Rv32Sha2Opcode::SHA512.local_usize()
        );
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

        // need to pad with one 1 bit, [C::MESSAGE_LENGTH_BITS] bits for the message length and then pad until the length is divisible by [C::BLOCK_BITS]
        let num_blocks = ((len << 3) as usize + 1 + C::MESSAGE_LENGTH_BITS).div_ceil(C::BLOCK_BITS);

        // we will read [num_blocks] * [C::BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(src as usize + num_blocks * C::BLOCK_CELLS <= (1 << self.air.ptr_max_bits));

        let hash_result: HashResult;
        if C::OPCODE_NAME == "SHA256" {
            let hasher = Sha256::new();
            hash_result =
                execute_hash::<F, C, Sha256>(hasher, num_blocks, src, len, dst, memory, e);
        } else if C::OPCODE_NAME == "SHA512" {
            let hasher = Sha512::new();
            hash_result =
                execute_hash::<F, C, Sha512>(hasher, num_blocks, src, len, dst, memory, e);
        } else {
            panic!("Unsupported opcode: {}", C::OPCODE_NAME);
        }

        self.records.push(Sha2Record {
            from_state: from_state.map(F::from_canonical_u32),
            dst_read,
            src_read,
            len_read,
            input_records: hash_result.input_records,
            input_message: hash_result.input_message,
            digest_writes: hash_result.digest_writes,
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

struct HashResult {
    input_records: Vec<Vec<RecordId>>,
    input_message: Vec<Vec<Vec<u8>>>,
    digest_writes: Vec<RecordId>,
}
fn execute_hash<F: PrimeField32, C: ShaChipConfig, H: Digest>(
    mut hasher: H,
    num_blocks: usize,
    src: u32,
    len: u32,
    dst: u32,
    memory: &mut MemoryController<F>,
    address_space: F,
) -> HashResult {
    let mut input_records: Vec<Vec<RecordId>> = Vec::with_capacity(num_blocks * C::NUM_READ_ROWS);
    let mut input_message = Vec::with_capacity(num_blocks * C::NUM_READ_ROWS);
    let mut read_ptr = src;
    for _ in 0..num_blocks {
        let block_reads_records = (0..C::NUM_READ_ROWS)
            .map(|i| {
                if C::OPCODE_NAME == "SHA256" {
                    let (id, data) = memory.read::<{ Sha256Config::READ_SIZE }>(
                        address_space,
                        F::from_canonical_u32(read_ptr + (i * C::READ_SIZE) as u32),
                    );
                    (id, data.to_vec())
                } else if C::OPCODE_NAME == "SHA512" {
                    let (id, data) = memory.read::<{ Sha512Config::READ_SIZE }>(
                        address_space,
                        F::from_canonical_u32(read_ptr + (i * C::READ_SIZE) as u32),
                    );
                    (id, data.to_vec())
                } else {
                    panic!("unsupported opcode: {}", C::OPCODE_NAME);
                }
            })
            .collect::<Vec<_>>();
        let block_reads_bytes = (0..C::NUM_READ_ROWS)
            .map(|i| {
                // we add to the hasher only the bytes that are part of the message
                let num_reads = min(C::READ_SIZE, (max(read_ptr, src + len) - read_ptr) as usize);
                let row_input: &[u8] = &block_reads_records[i]
                    .1
                    .iter()
                    .map(|x| x.as_canonical_u32().try_into().unwrap())
                    .collect::<Vec<_>>();
                hasher.update(&row_input[..num_reads]);
                read_ptr += C::READ_SIZE as u32;
                row_input.to_vec()
            })
            .collect::<Vec<_>>();
        input_records.push(block_reads_records.into_iter().map(|x| x.0).collect());
        input_message.push(block_reads_bytes);
    }

    let mut digest = vec![0u8; C::DIGEST_SIZE];
    digest.copy_from_slice(hasher.finalize().as_ref());

    match C::VARIANT {
        Sha2Variant::Sha256 => {
            debug_assert_eq!(C::NUM_WRITES, 1);
            let (digest_write, _) = memory.write::<{ Sha256Config::DIGEST_SIZE }>(
                address_space,
                F::from_canonical_u32(dst),
                digest
                    .iter()
                    .map(|b| F::from_canonical_u8(*b))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            );
            HashResult {
                input_records,
                input_message,
                digest_writes: vec![digest_write],
            }
        }
        Sha2Variant::Sha512 => {
            debug_assert_eq!(C::NUM_WRITES, 2);
            // write the digest in two halves because we only support writes up to 32 bytes
            let digest = digest
                .iter()
                .map(|b| F::from_canonical_u8(*b))
                .collect::<Vec<_>>();
            let mut digest_writes = Vec::with_capacity(C::NUM_WRITES);
            for i in 0..C::NUM_WRITES {
                let (digest_write, _) = memory.write::<{ Sha512Config::WRITE_SIZE }>(
                    address_space,
                    F::from_canonical_usize(dst as usize + i * Sha512Config::WRITE_SIZE),
                    digest[i * Sha512Config::WRITE_SIZE..(i + 1) * Sha512Config::WRITE_SIZE]
                        .try_into()
                        .unwrap(),
                );
                digest_writes.push(digest_write);
            }
            HashResult {
                input_records,
                input_message,
                digest_writes,
            }
        }
    }
}

// Returns a Vec of length C::DIGEST_SIZE
pub fn sha2_solve<C: ShaChipConfig>(input_message: &[u8]) -> Vec<u8> {
    match C::VARIANT {
        Sha2Variant::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
        Sha2Variant::Sha512 => {
            let mut hasher = Sha512::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
    }
}
