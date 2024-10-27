use std::sync::Arc;

use afs_compiler::conversion::CompilerOptions;
use ax_sdk::{
    afs_stark_backend::{config::StarkGenericConfig, keygen::types::MultiStarkProvingKey},
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use stark_vm::{arch::VmConfig, system::program::trace::CommittedProgram};

use crate::verifier::leaf::LeafVmVerifierConfig;

#[derive(Clone, Debug)]
pub struct AxiomVmConfig {
    pub poseidon2_max_constraint_degree: usize,
    pub max_num_user_public_values: usize,
    pub fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub compiler_options: CompilerOptions,
}

pub struct AxiomVmProvingKey {
    pub fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub app_vm_pk: MultiStarkProvingKey<BabyBearPoseidon2Config>,
    pub leaf_vm_config: VmConfig,
    pub leaf_verifier_pk: MultiStarkProvingKey<BabyBearPoseidon2Config>,
    pub committed_leaf_program: Arc<CommittedProgram<BabyBearPoseidon2Config>>,
}

impl AxiomVmProvingKey {
    pub fn keygen(config: AxiomVmConfig) -> Self {
        let engine = BabyBearPoseidon2Engine::new(config.fri_params);
        let app_vm_pk = config.app_vm_config.generate_pk(engine.keygen_builder());
        assert!(app_vm_pk.max_constraint_degree < 1 << config.fri_params.log_blowup);
        assert!(config.poseidon2_max_constraint_degree < 1 << config.fri_params.log_blowup);
        let leaf_vm_config = config.leaf_verifier_vm_config();
        let leaf_verifier_pk = leaf_vm_config.generate_pk(engine.keygen_builder());
        let leaf_program = LeafVmVerifierConfig {
            max_num_user_public_values: config.max_num_user_public_values,
            fri_params: config.fri_params,
            app_vm_config: config.app_vm_config.clone(),
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(app_vm_pk.get_vk());
        let committed_leaf_program =
            Arc::new(CommittedProgram::commit(&leaf_program, engine.config.pcs()));
        Self {
            fri_params: config.fri_params,
            app_vm_config: config.app_vm_config,
            app_vm_pk,
            leaf_vm_config,
            leaf_verifier_pk,
            committed_leaf_program,
        }
    }
}
