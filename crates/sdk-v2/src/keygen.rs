use std::sync::Arc;

use itertools::Itertools;
// use dummy::{compute_root_proof_heights, dummy_internal_proof_riscv_app_vm};
use openvm_circuit::{
    arch::{AirInventoryError, SystemConfig, VmCircuitConfig},
    system::memory::dimensions::MemoryDimensions,
};
use serde::{Deserialize, Serialize};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, StarkEngineV2,
    keygen::types::{
        MultiStarkProvingKeyV2 as MultiStarkProvingKey,
        MultiStarkVerifyingKeyV2 as MultiStarkVerifyingKey,
    },
    poseidon2::sponge::DuplexSponge,
};

use crate::{SC, config::AppConfig, prover::vm::types::VmProvingKey};

/// This is lightweight to clone as it contains smart pointers to the proving keys.
#[derive(Clone, Serialize, Deserialize)]
pub struct AppProvingKey<VC> {
    pub app_vm_pk: Arc<VmProvingKey<VC>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AppVerifyingKey {
    pub vk: MultiStarkVerifyingKey,
    pub memory_dimensions: MemoryDimensions,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AggProvingKey {
    pub leaf_pk: Arc<MultiStarkProvingKey>,
    pub internal_for_leaf_pk: Arc<MultiStarkProvingKey>,
    pub internal_recursive_pk: Arc<MultiStarkProvingKey>,
}

impl<VC> AppProvingKey<VC>
where
    VC: Clone + VmCircuitConfig<SC> + AsRef<SystemConfig>,
{
    pub fn keygen(config: AppConfig<VC>) -> Result<Self, AirInventoryError> {
        let app_engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(config.system_params);
        let app_vm_pk = {
            let vm_pk = app_engine
                .keygen(
                    &config
                        .app_vm_config
                        .create_airs()?
                        .into_airs()
                        .collect_vec(),
                )
                .0;
            VmProvingKey {
                vm_config: config.app_vm_config.clone(),
                vm_pk: Arc::new(vm_pk),
            }
        };
        Ok(Self {
            app_vm_pk: Arc::new(app_vm_pk),
        })
    }

    pub fn num_public_values(&self) -> usize {
        self.app_vm_pk.vm_config.as_ref().num_public_values
    }

    pub fn get_app_vk(&self) -> AppVerifyingKey {
        AppVerifyingKey {
            vk: self.app_vm_pk.vm_pk.get_vk(),
            memory_dimensions: self
                .app_vm_pk
                .vm_config
                .as_ref()
                .memory_config
                .memory_dimensions(),
        }
    }

    pub fn vm_config(&self) -> &VC {
        &self.app_vm_pk.vm_config
    }

    pub fn app_config(&self) -> AppConfig<VC> {
        AppConfig {
            app_vm_config: self.vm_config().clone(),
            system_params: self.app_vm_pk.vm_pk.params,
        }
    }
}
