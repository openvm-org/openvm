use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{config::StarkGenericConfig, keygen::types::MultiStarkProvingKey},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        baby_bear_poseidon2_outer::BabyBearPoseidon2OuterEngine,
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use axvm_circuit::{arch::VmConfig, system::program::trace::AxVmCommittedExe};
use axvm_native_compiler::conversion::CompilerOptions;

use crate::{
    verifier::{
        internal::InternalVmVerifierConfig, leaf::LeafVmVerifierConfig, root::RootVmVerifierConfig,
    },
    OuterSC,
};

type SC = BabyBearPoseidon2Config;

#[derive(Clone, Debug)]
pub struct AxiomVmConfig {
    pub max_num_user_public_values: usize,
    pub app_fri_params: FriParameters,
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub root_fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub compiler_options: CompilerOptions,
}

// TODO: separate the Agg VM part out.
pub struct AxiomVmProvingKey {
    pub app_fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub app_vm_pk: MultiStarkProvingKey<SC>,
    pub leaf_fri_params: FriParameters,
    pub leaf_vm_config: VmConfig,
    pub leaf_vm_pk: MultiStarkProvingKey<SC>,
    pub internal_fri_params: FriParameters,
    pub internal_vm_config: VmConfig,
    pub internal_vm_pk: MultiStarkProvingKey<SC>,
    pub committed_leaf_program: Arc<AxVmCommittedExe<SC>>,
    pub committed_internal_program: Arc<AxVmCommittedExe<SC>>,
    pub root_fri_params: FriParameters,
    pub root_vm_config: VmConfig,
    pub root_vm_pk: MultiStarkProvingKey<OuterSC>,
    pub committed_root_program: Arc<AxVmCommittedExe<OuterSC>>,
}

impl AxiomVmProvingKey {
    pub fn keygen(config: AxiomVmConfig) -> Self {
        let app_engine = BabyBearPoseidon2Engine::new(config.app_fri_params);
        let app_vm_pk = config
            .app_vm_config
            .generate_pk(app_engine.keygen_builder());
        assert!(app_vm_pk.max_constraint_degree <= config.app_fri_params.max_constraint_degree());
        assert_eq!(
            config.max_num_user_public_values,
            config.app_vm_config.num_public_values
        );
        assert!(config.app_vm_config.continuation_enabled);
        let leaf_engine = BabyBearPoseidon2Engine::new(config.leaf_fri_params);
        let leaf_vm_config = config.leaf_vm_config();
        let leaf_vm_pk = leaf_vm_config.generate_pk(leaf_engine.keygen_builder());
        assert!(leaf_vm_pk.max_constraint_degree <= config.leaf_fri_params.max_constraint_degree());
        let leaf_vm_vk = leaf_vm_pk.get_vk();
        let leaf_program = LeafVmVerifierConfig {
            app_fri_params: config.app_fri_params,
            app_vm_config: config.app_vm_config.clone(),
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&app_vm_pk.get_vk());
        let committed_leaf_program = Arc::new(AxVmCommittedExe::commit(
            leaf_program.into(),
            leaf_engine.config.pcs(),
        ));

        let internal_engine = BabyBearPoseidon2Engine::new(config.internal_fri_params);
        let internal_vm_config = config.internal_vm_config();
        let internal_vm_pk = internal_vm_config.generate_pk(internal_engine.keygen_builder());
        assert!(
            internal_vm_pk.max_constraint_degree
                <= config.internal_fri_params.max_constraint_degree()
        );
        let internal_vm_vk = internal_vm_pk.get_vk();
        let internal_program = InternalVmVerifierConfig {
            leaf_fri_params: config.leaf_fri_params,
            internal_fri_params: config.internal_fri_params,
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&leaf_vm_vk, &internal_vm_vk);
        let committed_internal_program = Arc::new(AxVmCommittedExe::<SC>::commit(
            internal_program.into(),
            internal_engine.config.pcs(),
        ));

        let root_engine = BabyBearPoseidon2OuterEngine::new(config.root_fri_params);
        let root_vm_config = config.root_verifier_vm_config();
        let root_vm_pk = root_vm_config.generate_pk(root_engine.keygen_builder());
        assert!(root_vm_pk.max_constraint_degree <= config.root_fri_params.max_constraint_degree());
        let root_program = RootVmVerifierConfig {
            leaf_fri_params: config.leaf_fri_params,
            internal_fri_params: config.internal_fri_params,
            num_public_values: config.max_num_user_public_values,
            internal_vm_verifier_commit: committed_internal_program
                .committed_program
                .prover_data
                .commit
                .into(),
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&leaf_vm_vk, &internal_vm_vk);
        let committed_root_program = Arc::new(AxVmCommittedExe::<OuterSC>::commit(
            root_program.into(),
            root_engine.config.pcs(),
        ));
        Self {
            app_fri_params: config.app_fri_params,
            app_vm_config: config.app_vm_config,
            app_vm_pk,
            leaf_fri_params: config.leaf_fri_params,
            leaf_vm_config,
            leaf_vm_pk,
            internal_fri_params: config.internal_fri_params,
            internal_vm_config,
            internal_vm_pk,
            committed_leaf_program,
            committed_internal_program,
            root_fri_params: config.root_fri_params,
            root_vm_config,
            root_vm_pk,
            committed_root_program,
        }
    }
}
