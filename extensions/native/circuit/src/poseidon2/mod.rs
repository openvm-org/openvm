use crate::poseidon2::air::{NativePoseidon2Air, VerifyBatchBus};
use crate::poseidon2::chip::NativePoseidon2Step;
use openvm_circuit::arch::{ExecutionBridge, NewVmChipWrapper, Streams, SystemPort};
use openvm_circuit::system::memory::SharedMemoryHelper;
use openvm_native_compiler::conversion::AS;
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir};
use openvm_stark_backend::p3_field::PrimeField32;
use std::sync::{Arc, Mutex};

pub mod air;
pub mod chip;
mod columns;
#[cfg(test)]
mod tests;

const CHUNK: usize = 8;
pub type NativePoseidon2Chip<F, const SBOX_REGISTERS: usize> = NewVmChipWrapper<
    F,
    NativePoseidon2Air<F, SBOX_REGISTERS>,
    NativePoseidon2Step<F, SBOX_REGISTERS>,
>;

pub fn new_native_poseidon2_chip<F: PrimeField32, const SBOX_REGISTERS: usize>(
    port: SystemPort,
    poseidon2_config: Poseidon2Config<F>,
    verify_batch_bus: VerifyBatchBus,
    streams: Arc<Mutex<Streams<F>>>,
    max_ins_capacity: usize,
    mem_helper: SharedMemoryHelper<F>,
) -> NativePoseidon2Chip<F, SBOX_REGISTERS> {
    NativePoseidon2Chip::<F, SBOX_REGISTERS>::new(
        NativePoseidon2Air {
            execution_bridge: ExecutionBridge::new(port.execution_bus, port.program_bus),
            memory_bridge: port.memory_bridge,
            internal_bus: verify_batch_bus,
            subair: Arc::new(Poseidon2SubAir::new(poseidon2_config.constants.into())),
            address_space: F::from_canonical_u32(AS::Native as u32),
        },
        NativePoseidon2Step::new(poseidon2_config, streams),
        max_ins_capacity,
        mem_helper,
    )
}
