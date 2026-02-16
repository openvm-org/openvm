use cuda_backend_v2::{
    BabyBearPoseidon2GpuEngineV2 as GpuBabyBearPoseidon2Engine, GpuBackendV2 as GpuBackend,
};
use openvm_circuit::arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension};
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
