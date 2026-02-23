use std::sync::Arc;

use openvm_circuit::{
    arch::{
        hasher::{Hasher, HasherChip},
        VmField,
    },
    system::poseidon2::{
        air::Poseidon2PeripheryAir, Poseidon2PeripheryBaseChip, PERIPHERY_POSEIDON2_CHUNK_SIZE,
    },
};
use openvm_circuit_primitives::Chip;
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir};
use openvm_stark_backend::{
    interaction::LookupBus,
    prover::{AirProvingContext, CpuBackend},
    StarkProtocolConfig, Val,
};

pub mod bus;

const SBOX_REGISTERS: usize = 1;
pub type DeferralPoseidon2Air<F> = Poseidon2PeripheryAir<F, SBOX_REGISTERS>;

#[derive(Debug)]
pub struct DeferralPoseidon2Chip<F: VmField>(Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>);

impl<F: VmField> DeferralPoseidon2Chip<F> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        Self(Poseidon2PeripheryBaseChip::new(poseidon2_config))
    }
}

impl<F: VmField> Hasher<PERIPHERY_POSEIDON2_CHUNK_SIZE, F> for DeferralPoseidon2Chip<F> {
    fn compress(
        &self,
        lhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        rhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    ) -> [F; PERIPHERY_POSEIDON2_CHUNK_SIZE] {
        self.0.compress(lhs, rhs)
    }
}

impl<F: VmField> HasherChip<PERIPHERY_POSEIDON2_CHUNK_SIZE, F> for DeferralPoseidon2Chip<F> {
    fn compress_and_record(
        &self,
        lhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        rhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    ) -> [F; PERIPHERY_POSEIDON2_CHUNK_SIZE] {
        self.0.compress_and_record(lhs, rhs)
    }
}

impl<RA, SC: StarkProtocolConfig> Chip<RA, CpuBackend<SC>> for DeferralPoseidon2Chip<Val<SC>>
where
    Val<SC>: VmField,
{
    fn generate_proving_ctx(&self, records: RA) -> AirProvingContext<CpuBackend<SC>> {
        self.0.generate_proving_ctx(records)
    }
}

pub fn deferral_poseidon2_air<F: VmField>(bus: LookupBus) -> DeferralPoseidon2Air<F> {
    let constants = Poseidon2Config::default().constants;
    let subair = Arc::new(Poseidon2SubAir::new(constants.into()));
    DeferralPoseidon2Air::new(subair, bus)
}

pub fn deferral_poseidon2_chip<F: VmField>() -> DeferralPoseidon2Chip<F> {
    let config = Poseidon2Config::default();
    DeferralPoseidon2Chip::new(config)
}
