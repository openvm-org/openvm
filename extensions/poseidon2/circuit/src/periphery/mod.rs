use openvm_circuit::arch::VmField;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::interaction::LookupBus;

use crate::SBOX_REGISTERS;

mod air;
mod trace;

#[cfg(test)]
pub mod tests;

pub use air::*;
pub use trace::*;

pub fn poseidon2_periphery_air<F: VmField>(bus: LookupBus) -> Poseidon2PeripheryAir<F> {
    let config = Poseidon2Config::default();
    Poseidon2PeripheryAir::new(config, bus)
}

pub fn poseidon2_periphery_chip<F: VmField>() -> Poseidon2PeripheryChip<F> {
    let config = Poseidon2Config::default();
    Poseidon2PeripheryChip::new(config)
}
