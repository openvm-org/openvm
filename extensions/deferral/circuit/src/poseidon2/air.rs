use std::sync::Arc;

use openvm_circuit::{arch::VmField, system::poseidon2::air::Poseidon2PeripheryAir};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir};
use openvm_stark_backend::interaction::LookupBus;

use super::SBOX_REGISTERS;

pub type DeferralPoseidon2Air<F> = Poseidon2PeripheryAir<F, SBOX_REGISTERS>;

pub fn deferral_poseidon2_air<F: VmField>(bus: LookupBus) -> DeferralPoseidon2Air<F> {
    let constants = Poseidon2Config::default().constants;
    let subair = Arc::new(Poseidon2SubAir::new(constants.into()));
    DeferralPoseidon2Air::new(subair, bus)
}
