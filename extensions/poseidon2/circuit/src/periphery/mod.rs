use openvm_circuit::arch::VmField;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::interaction::LookupBus;

mod air;
mod trace;

#[cfg(test)]
pub mod tests;

pub use air::*;
pub use trace::*;

const SBOX_REGISTERS: usize = 1;

pub fn poseidon2_periphery_air<F: VmField>(bus: LookupBus) -> Poseidon2PermuteAir<F> {
    let config = Poseidon2Config::default();
    Poseidon2PermuteAir::new(config, bus)
}

pub fn poseidon2_periphery_chip<F: VmField>() -> Poseidon2PermuteChip<F> {
    let config = Poseidon2Config::default();
    Poseidon2PermuteChip::new(config)
}
