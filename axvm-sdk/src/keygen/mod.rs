use std::sync::Arc;

use ax_stark_sdk::{
    ax_stark_backend::{
        config::{Com, StarkGenericConfig},
        prover::types::Proof,
        Chip,
    },
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        baby_bear_poseidon2_outer::BabyBearPoseidon2OuterEngine,
    },
    engine::StarkFriEngine,
};
use axvm_circuit::{
    arch::{new_vm::VirtualMachine, VmGenericConfig},
    prover::types::VmProvingKey,
    system::program::trace::AxVmCommittedExe,
};
use axvm_native_circuit::NativeConfig;
use axvm_native_compiler::ir::DIGEST_SIZE;
use derivative::Derivative;
use dummy::{compute_root_proof_heights, dummy_internal_proof_riscv_app_vm};
use serde::{Deserialize, Serialize};

use crate::{
    config::{AggConfig, AppConfig},
    keygen::perm::AirIdPermutation,
    verifier::{internal::InternalVmVerifierConfig, root::RootVmVerifierConfig},
    OuterSC, F, SC,
};

pub(crate) mod dummy;
pub mod perm;

#[derive(Clone, Serialize, Deserialize)]
pub struct AppProvingKey<VmConfig> {
    pub app_vm_pk: VmProvingKey<SC, VmConfig>,
}

impl<VmConfig: VmGenericConfig<F>> AppProvingKey<VmConfig>
where
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    pub fn keygen(config: AppConfig<VmConfig>) -> Self {
        let app_engine = BabyBearPoseidon2Engine::new(config.app_fri_params);
        let app_vm_pk = {
            let vm = VirtualMachine::new(app_engine, config.app_vm_config.clone());
            let vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.app_fri_params.max_constraint_degree());
            assert!(config.app_vm_config.system().continuation_enabled);
            VmProvingKey {
                fri_params: config.app_fri_params,
                vm_config: config.app_vm_config.clone(),
                vm_pk,
            }
        };
        Self { app_vm_pk }
    }

    pub fn num_public_values(&self) -> usize {
        self.app_vm_pk.vm_config.system().num_public_values
    }
}

#[derive(Serialize, Deserialize)]
pub struct AggProvingKey {
    pub leaf_vm_pk: VmProvingKey<SC, NativeConfig>,
    pub internal_vm_pk: VmProvingKey<SC, NativeConfig>,
    pub internal_committed_exe: Arc<AxVmCommittedExe<SC>>,
    pub root_verifier_pk: RootVerifierProvingKey,
}

impl AggProvingKey {
    pub fn keygen(config: AggConfig) -> Self {
        Self::dummy_proof_and_keygen(config).0
    }

    pub fn dummy_proof_and_keygen(config: AggConfig) -> (Self, Proof<SC>) {
        let leaf_vm_config = config.leaf_vm_config();
        let internal_vm_config = config.internal_vm_config();
        let root_vm_config = config.root_verifier_vm_config();

        let leaf_engine = BabyBearPoseidon2Engine::new(config.leaf_fri_params);
        let leaf_vm_pk = {
            let vm = VirtualMachine::new(leaf_engine, leaf_vm_config.clone());
            let vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.leaf_fri_params.max_constraint_degree());
            VmProvingKey {
                fri_params: config.leaf_fri_params,
                vm_config: leaf_vm_config,
                vm_pk,
            }
        };
        let leaf_vm_vk = leaf_vm_pk.vm_pk.get_vk();

        let internal_engine = BabyBearPoseidon2Engine::new(config.internal_fri_params);
        let internal_vm = VirtualMachine::new(internal_engine, internal_vm_config.clone());
        let internal_vm_pk = {
            let vm_pk = internal_vm.keygen();
            assert!(
                vm_pk.max_constraint_degree <= config.internal_fri_params.max_constraint_degree()
            );
            VmProvingKey {
                fri_params: config.internal_fri_params,
                vm_config: internal_vm_config,
                vm_pk,
            }
        };
        let internal_vm_vk = internal_vm_pk.vm_pk.get_vk();

        let internal_program = InternalVmVerifierConfig {
            leaf_fri_params: config.leaf_fri_params,
            internal_fri_params: config.internal_fri_params,
            compiler_options: config.compiler_options.clone(),
        }
        .build_program(&leaf_vm_vk, &internal_vm_vk);
        let internal_committed_exe = Arc::new(AxVmCommittedExe::<SC>::commit(
            internal_program.into(),
            internal_vm.engine.config.pcs(),
        ));

        let internal_proof = dummy_internal_proof_riscv_app_vm(
            leaf_vm_pk.clone(),
            internal_vm_pk.clone(),
            internal_committed_exe.clone(),
            config.max_num_user_public_values,
        );

        let root_verifier_pk = {
            let root_engine = BabyBearPoseidon2OuterEngine::new(config.root_fri_params);
            let root_program = RootVmVerifierConfig {
                leaf_fri_params: config.leaf_fri_params,
                internal_fri_params: config.internal_fri_params,
                num_public_values: config.max_num_user_public_values,
                internal_vm_verifier_commit: internal_committed_exe.get_program_commit().into(),
                compiler_options: config.compiler_options,
            }
            .build_program(&leaf_vm_vk, &internal_vm_vk);
            let root_committed_exe = Arc::new(AxVmCommittedExe::<OuterSC>::commit(
                root_program.into(),
                root_engine.config.pcs(),
            ));

            let vm = VirtualMachine::new(root_engine, root_vm_config.clone());
            let mut vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.root_fri_params.max_constraint_degree());

            let (air_heights, _internal_heights) = compute_root_proof_heights(
                root_vm_config.clone(),
                root_committed_exe.exe.clone(),
                &internal_proof,
            );
            let root_air_perm = AirIdPermutation::compute(&air_heights);
            root_air_perm.permute(&mut vm_pk.per_air);

            RootVerifierProvingKey {
                vm_pk: VmProvingKey {
                    fri_params: config.root_fri_params,
                    vm_config: root_vm_config,
                    vm_pk,
                },
                root_committed_exe,
                air_heights,
            }
        };

        (
            Self {
                leaf_vm_pk,
                internal_vm_pk,
                internal_committed_exe,
                root_verifier_pk,
            },
            internal_proof,
        )
    }

    pub fn internal_program_commit(&self) -> [F; DIGEST_SIZE] {
        self.internal_committed_exe.get_program_commit().into()
    }

    pub fn num_public_values(&self) -> usize {
        self.root_verifier_pk
            .vm_pk
            .vm_config
            .system
            .num_public_values
            - (2 * DIGEST_SIZE)
    }
}

/// Proving key for the root verifier.
/// Properties:
/// - Traces heights of each AIR is constant. This is required by the static verifier.
/// - Instead of the AIR order specified by VmConfig. AIRs are ordered by trace heights.
#[derive(Serialize, Deserialize, Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct RootVerifierProvingKey {
    /// VM Proving key for the root verifier.
    /// - AIR proving key in `MultiStarkProvingKey` is ordered by trace height.
    /// - `VmConfig.overridden_executor_heights` is specified and is in the original AIR order.
    /// - `VmConfig.memory_config.boundary_air_height` is specified.
    pub vm_pk: VmProvingKey<OuterSC, NativeConfig>,
    /// Committed executable for the root VM.
    pub root_committed_exe: Arc<AxVmCommittedExe<OuterSC>>,
    /// The constant trace heights, ordered by AIR ID.
    pub air_heights: Vec<usize>,
    // The following is currently not used:
    // The constant trace heights, ordered according to an internal ordering determined by the `NativeConfig`.
    // pub internal_heights: VmComplexTraceHeights,
}

impl RootVerifierProvingKey {
    pub fn air_id_permutation(&self) -> AirIdPermutation {
        AirIdPermutation::compute(&self.air_heights)
    }
}
