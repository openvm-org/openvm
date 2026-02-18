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
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::{
    interaction::BusIndex,
    prover::{AirProvingContext, CpuBackend},
    StarkProtocolConfig, Val,
};

pub mod bus;
use bus::*;

const SBOX_REGISTERS: usize = 1;
pub type DeferralPoseidon2Air<F> = Poseidon2PeripheryAir<F, SBOX_REGISTERS>;

#[derive(Debug)]
pub struct DeferralPoseidon2Chip<F: VmField>(Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>);

impl<F: VmField> DeferralPoseidon2Chip<F> {
    pub fn new(poseidon2_config: Poseidon2Config<F>, bus_idx: BusIndex) -> Self {
        Self(Poseidon2PeripheryBaseChip::new(poseidon2_config, bus_idx))
    }

    pub fn air(&self) -> Arc<DeferralPoseidon2Air<F>> {
        self.0.air.clone()
    }

    pub fn bus(&self) -> DeferralPoseidon2Bus {
        DeferralPoseidon2Bus(self.0.air.bus)
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
