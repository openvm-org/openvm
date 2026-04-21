use openvm_circuit::arch::VmField;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::interaction::LookupBus;

mod air;
mod bus;
#[cfg(feature = "cuda")]
mod cuda;
mod trace;

#[cfg(test)]
pub mod tests;

pub use air::*;
pub use bus::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
pub use trace::*;

const SBOX_REGISTERS: usize = 1;

pub fn deferral_poseidon2_air<F: VmField>(bus: LookupBus) -> DeferralPoseidon2Air<F> {
    let config = Poseidon2Config::default();
    DeferralPoseidon2Air::new(config, bus)
}

pub fn deferral_poseidon2_chip<F: VmField>() -> DeferralPoseidon2Chip<F> {
    let config = Poseidon2Config::default();
    DeferralPoseidon2Chip::new(config)
}
