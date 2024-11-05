use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, keygen::types::MultiStarkProvingKey},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axvm_circuit::{arch::VmConfig, system::program::trace::AxVmCommittedExe};
use axvm_native_compiler::conversion::CompilerOptions;

use crate::verifier::{
    internal::InternalVmVerifierConfig, leaf::LeafVmVerifierConfig, root::RootVmVerifierConfig,
};

type SC = BabyBearPoseidon2Config;

#[derive(Clone, Debug)]
pub struct AxiomVmConfig {
    pub poseidon2_max_constraint_degree: usize,
    pub max_num_user_public_values: usize,
    pub fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub compiler_options: CompilerOptions,
}

// TODO: separate the Agg VM part out.
pub struct AxiomVmProvingKey {
    pub fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub app_vm_pk: MultiStarkProvingKey<SC>,
    pub non_root_agg_vm_config: VmConfig,
    pub non_root_agg_vm_pk: MultiStarkProvingKey<SC>,
    pub committed_leaf_program: Arc<AxVmCommittedExe<SC>>,
    pub committed_internal_program: Arc<AxVmCommittedExe<SC>>,
    pub root_agg_vm_config: VmConfig,
    pub root_agg_vm_pk: MultiStarkProvingKey<SC>,
    pub committed_root_program: Arc<AxVmCommittedExe<SC>>,
}

impl AxiomVmProvingKey {
    pub fn keygen(config: AxiomVmConfig) -> Self {
        let engine = BabyBearPoseidon2Engine::new(config.fri_params);
        let app_vm_pk = config.app_vm_config.generate_pk(engine.keygen_builder());
        assert!(app_vm_pk.max_constraint_degree < 1 << config.fri_params.log_blowup);
        assert!(config.poseidon2_max_constraint_degree < 1 << config.fri_params.log_blowup);
        assert_eq!(
            config.max_num_user_public_values,
            config.app_vm_config.num_public_values
        );
        assert!(config.app_vm_config.continuation_enabled);
        let non_root_agg_vm_config = config.non_root_verifier_vm_config();
        let non_root_agg_vm_pk = non_root_agg_vm_config.generate_pk(engine.keygen_builder());
        let non_root_agg_vm_vk = non_root_agg_vm_pk.get_vk();
        let leaf_program = LeafVmVerifierConfig {
            fri_params: config.fri_params,
            app_vm_config: config.app_vm_config.clone(),
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&app_vm_pk.get_vk());
        let committed_leaf_program = Arc::new(AxVmCommittedExe::commit(
            leaf_program.into(),
            engine.config.pcs(),
        ));
        let internal_program = InternalVmVerifierConfig {
            fri_params: config.fri_params,
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&non_root_agg_vm_vk);
        let committed_internal_program = Arc::new(AxVmCommittedExe::<SC>::commit(
            internal_program.into(),
            engine.config.pcs(),
        ));

        let root_agg_vm_config = config.root_verifier_vm_config();
        let root_agg_vm_pk = root_agg_vm_config.generate_pk(engine.keygen_builder());
        let root_program = RootVmVerifierConfig {
            fri_params: config.fri_params,
            num_public_values: config.max_num_user_public_values,
            internal_vm_verifier_commit: committed_internal_program
                .committed_program
                .prover_data
                .commit
                .into(),
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&non_root_agg_vm_vk);
        let committed_root_program = Arc::new(AxVmCommittedExe::<SC>::commit(
            root_program.into(),
            engine.config.pcs(),
        ));
        Self {
            fri_params: config.fri_params,
            app_vm_config: config.app_vm_config,
            app_vm_pk,
            non_root_agg_vm_config,
            non_root_agg_vm_pk,
            committed_leaf_program,
            committed_internal_program,
            root_agg_vm_config,
            root_agg_vm_pk,
            committed_root_program,
        }
    }
}
