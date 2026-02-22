use openvm_circuit::arch::{
    ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use super::DeferralExtension;

pub struct DeferralGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, DeferralExtension>
    for DeferralGpuProverExt
{
    fn extend_prover(
        &self,
        _: &DeferralExtension,
        _inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        Ok(())
    }
}
