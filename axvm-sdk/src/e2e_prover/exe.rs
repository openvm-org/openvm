use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::config::{StarkGenericConfig, Val},
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::StarkFriEngine,
};
use axvm_circuit::{arch::instructions::exe::AxVmExe, system::program::trace::AxVmCommittedExe};

use crate::{
    config::{AggConfig, AppConfig},
    keygen::AppProvingKey,
    verifier::leaf::LeafVmVerifierConfig,
    SC,
};

pub fn commit_app_exe(
    app_config: AppConfig,
    app_exe: impl Into<AxVmExe<Val<SC>>>,
) -> Arc<AxVmCommittedExe<SC>> {
    let mut exe: AxVmExe<_> = app_exe.into();
    exe.program.max_num_public_values = app_config.app_vm_config.num_public_values;
    let app_engine = BabyBearPoseidon2Engine::new(app_config.app_fri_params);
    Arc::new(AxVmCommittedExe::<SC>::commit(exe, app_engine.config.pcs()))
}

pub fn generate_leaf_committed_exe(
    agg_config: AggConfig,
    app_pk: &AppProvingKey,
) -> Arc<AxVmCommittedExe<SC>> {
    let app_vm_vk = app_pk.app_vm_pk.vm_pk.get_vk();
    let leaf_engine = BabyBearPoseidon2Engine::new(agg_config.leaf_fri_params);
    let leaf_program = LeafVmVerifierConfig {
        app_fri_params: app_pk.app_vm_pk.fri_params,
        app_vm_config: app_pk.app_vm_pk.vm_config.clone(),
        compiler_options: agg_config.compiler_options,
    }
    .build_program(&app_vm_vk);
    Arc::new(AxVmCommittedExe::commit(
        leaf_program.into(),
        leaf_engine.config.pcs(),
    ))
}
