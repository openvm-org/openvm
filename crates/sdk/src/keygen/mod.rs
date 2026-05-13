use std::sync::Arc;

use itertools::Itertools;
use openvm_circuit::{
    arch::{AirInventoryError, SystemConfig, VmCircuitConfig},
    system::memory::dimensions::MemoryDimensions,
};
#[cfg(feature = "root-prover")]
use openvm_continuations::RootSC;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    AirRef, StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge};
use serde::{Deserialize, Serialize};

use crate::{config::AppConfig, prover::vm::types::VmProvingKey, SC};

#[cfg(feature = "root-prover")]
pub mod dummy;
#[cfg(feature = "evm-prove")]
pub mod static_verifier;

/// This is lightweight to clone as it contains smart pointers to the proving keys.
#[derive(Clone, Serialize, Deserialize)]
pub struct AppProvingKey<VC> {
    pub app_vm_pk: Arc<VmProvingKey<VC>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AppVerifyingKey {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub memory_dimensions: MemoryDimensions,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AggPrefixProvingKey {
    pub leaf: Arc<MultiStarkProvingKey<SC>>,
    pub internal_for_leaf: Arc<MultiStarkProvingKey<SC>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AggProvingKey {
    pub prefix: AggPrefixProvingKey,
    pub internal_recursive: Arc<MultiStarkProvingKey<SC>>,
}

#[cfg(feature = "root-prover")]
#[derive(Clone, Serialize, Deserialize)]
pub struct RootProvingKey {
    pub root_pk: Arc<MultiStarkProvingKey<RootSC>>,
    pub trace_heights: Vec<usize>,
}

impl<VC> AppProvingKey<VC>
where
    VC: Clone + VmCircuitConfig<SC> + AsRef<SystemConfig>,
{
    pub fn keygen(config: AppConfig<VC>) -> Result<Self, AirInventoryError> {
        let app_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(config.system_params);
        let app_vm_pk = {
            let vm_pk = app_engine
                .keygen(
                    &config
                        .app_vm_config
                        .create_airs()?
                        .into_airs()
                        .map(|a| a as AirRef<_>)
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
            system_params: self.app_vm_pk.vm_pk.params.clone(),
        }
    }
}

/// Attention: the serialized size of this struct is VERY large, usually >10GB.
#[cfg(feature = "evm-prove")]
#[derive(Clone, Serialize, Deserialize)]
pub struct Halo2ProvingKey {
    /// Static verifier to verify a stark proof of the root verifier.
    pub verifier: Arc<openvm_static_verifier::StaticVerifierProvingKey>,
    /// Wrapper circuit to verify static verifier and reduce the verification costs in the final
    /// proof.
    pub wrapper: Arc<openvm_static_verifier::Halo2WrapperProvingKey>,
    /// Whether to collect detailed profiling metrics.
    pub profiling: bool,
}
