use std::{path::PathBuf, sync::Arc};

use derivative::Derivative;
use openvm_circuit::{
    arch::{VirtualMachine, VmConfig},
    system::program::trace::VmCommittedExe,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_native_recursion::halo2::{
    utils::Halo2ParamsReader, verifier::Halo2VerifierProvingKey, wrapper::Halo2WrapperProvingKey,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        baby_bear_poseidon2_root::BabyBearPoseidon2RootEngine, FriParameters,
    },
    engine::StarkFriEngine,
    openvm_stark_backend::{
        config::{Com, StarkGenericConfig},
        keygen::types::MultiStarkVerifyingKey,
        prover::types::Proof,
        Chip,
    },
    p3_bn254_fr::Bn254Fr,
};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::{
    commit::babybear_digest_to_bn254,
    config::{AggConfig, AggStarkConfig, AppConfig, MinimalConfig, MinimalStarkConfig},
    keygen::{
        dummy::{
            compute_minimal_root_proof_heights, compute_root_proof_heights,
            dummy_internal_proof_riscv_app_vm, dummy_minimal_proof,
        },
        perm::AirIdPermutation,
        AppVerifyingKey, Halo2ProvingKey, RootVerifierProvingKey,
    },
    prover::vm::types::VmProvingKey,
    static_verifier::StaticVerifierPvHandler,
    verifier::{
        internal::InternalVmVerifierConfig, leaf::LeafVmVerifierConfig,
        minimal::MinimalVmVerifierConfig, root::RootVmVerifierConfig,
    },
    NonRootCommittedExe, RootSC, F, SC,
};

/// Minimal proving key for App->Root->Static Verifier->Wrapper
#[derive(Clone, Serialize, Deserialize)]
pub struct MinimalProvingKey<VC> {
    pub minimal_stark_pk: MinimalStarkProvingKey<VC>,
    pub halo2_pk: Halo2ProvingKey,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MinimalStarkProvingKey<VC> {
    pub app_vm_pk: Arc<VmProvingKey<SC, VC>>,
    // pub app_committed_exe: Arc<NonRootCommittedExe>,
    pub root_verifier_pk: RootVerifierProvingKey,
}

impl<VC> MinimalStarkProvingKey<VC>
where
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn keygen(config: MinimalStarkConfig<VC>) -> Self {
        tracing::info_span!("minimal_stark_keygen", group = "minimal_stark_keygen")
            .in_scope(|| Self::dummy_proof_and_keygen(config).0)
    }

    pub fn dummy_proof_and_keygen(config: MinimalStarkConfig<VC>) -> (Self, Proof<SC>) {
        let root_vm_config = config.minimal_root_verifier_vm_config();

        println!(
            "MinimalStarkConfig app_fri_params: {:?}",
            config.app_fri_params
        );
        println!(
            "MinimalStarkConfig app_vm_config: {:?}",
            config.app_vm_config.system().clone()
        );

        let app_engine = BabyBearPoseidon2Engine::new(config.app_fri_params);
        let app_vm_pk = Arc::new({
            let vm = VirtualMachine::new(app_engine, config.app_vm_config.clone());
            let vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.app_fri_params.max_constraint_degree());
            VmProvingKey {
                fri_params: config.app_fri_params,
                vm_config: config.app_vm_config.clone(),
                vm_pk,
            }
        });

        let dummy_app_proof =
            dummy_minimal_proof(config.app_fri_params, config.max_num_user_public_values);

        let root_verifier_pk = {
            let root_engine = BabyBearPoseidon2RootEngine::new(config.root_fri_params);
            let minimal_root_program = MinimalVmVerifierConfig {
                num_public_values: config.max_num_user_public_values,
                app_fri_params: config.app_fri_params,
                app_system_config: config.app_vm_config.system().clone(),
                compiler_options: config.compiler_options,
            }
            .build_program(&app_vm_pk.vm_pk.get_vk());
            let minimal_root_committed_exe = Arc::new(VmCommittedExe::<RootSC>::commit(
                minimal_root_program.into(),
                root_engine.config.pcs(),
            ));

            let vm = VirtualMachine::new(root_engine, root_vm_config.clone());
            let mut vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.root_fri_params.max_constraint_degree());

            let (air_heights, _internal_heights) = compute_minimal_root_proof_heights(
                root_vm_config.clone(),
                minimal_root_committed_exe.exe.clone(),
                &dummy_app_proof,
            );
            let root_air_perm = AirIdPermutation::compute(&air_heights);
            root_air_perm.permute(&mut vm_pk.per_air);

            RootVerifierProvingKey {
                vm_pk: Arc::new(VmProvingKey {
                    fri_params: config.root_fri_params,
                    vm_config: root_vm_config,
                    vm_pk,
                }),
                root_committed_exe: minimal_root_committed_exe,
                air_heights,
            }
        };

        (
            Self {
                app_vm_pk: app_vm_pk.clone(),
                root_verifier_pk,
            },
            dummy_app_proof,
        )
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

/// Proving key for the minimal root verifier.
/// Properties:
/// - Traces heights of each AIR is constant. This is required by the static verifier.
/// - Instead of the AIR order specified by VC. AIRs are ordered by trace heights.
#[derive(Serialize, Deserialize, Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct MinimalRootVerifierProvingKey {
    /// VM Proving key for the root verifier.
    /// - AIR proving key in `MultiStarkProvingKey` is ordered by trace height.
    /// - `VmConfig.overridden_executor_heights` is specified and is in the original AIR order.
    /// - `VmConfig.memory_config.boundary_air_height` is specified.
    pub vm_pk: Arc<VmProvingKey<RootSC, NativeConfig>>,
    /// Committed executable for the root VM.
    pub root_committed_exe: Arc<VmCommittedExe<RootSC>>,
    /// The constant trace heights, ordered by AIR ID.
    pub air_heights: Vec<usize>,
    // The following is currently not used:
    // The constant trace heights, ordered according to an internal ordering determined by the `NativeConfig`.
    // pub internal_heights: VmComplexTraceHeights,
}

impl MinimalRootVerifierProvingKey {
    pub fn air_id_permutation(&self) -> AirIdPermutation {
        AirIdPermutation::compute(&self.air_heights)
    }
}

impl<VC> MinimalProvingKey<VC> {
    /// Attention:
    /// - This function is very expensive.
    /// - Please make sure SRS(KZG parameters) is already downloaded.
    #[tracing::instrument(level = "info", fields(group = "minimal_keygen"), skip_all)]
    pub fn keygen(minimal_config: MinimalConfig<VC>, reader: &impl Halo2ParamsReader) -> Self
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let MinimalConfig {
            minimal_stark_config,
            halo2_config,
        } = minimal_config;
        let (minimal_stark_pk, dummy_proof) =
            MinimalStarkProvingKey::dummy_proof_and_keygen(minimal_stark_config);
        let dummy_root_proof = minimal_stark_pk
            .root_verifier_pk
            .generate_dummy_minimal_root_proof(dummy_proof);

        // TODO: Remove; this is only for testing. cache the halo2 proving key to speed up test.
        let halo2_pk_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("halo2_pk.bin");
        let halo2_pk = if !halo2_pk_path.exists() {
            println!("Generating Halo2 proving key");
            let verifier = minimal_stark_pk.root_verifier_pk.keygen_static_verifier(
                &reader.read_params(halo2_config.verifier_k),
                dummy_root_proof,
                None::<&RootVerifierProvingKey>,
            );
            let dummy_snark = verifier.generate_dummy_snark(reader);
            let wrapper = if let Some(wrapper_k) = halo2_config.wrapper_k {
                Halo2WrapperProvingKey::keygen(&reader.read_params(wrapper_k), dummy_snark)
            } else {
                Halo2WrapperProvingKey::keygen_auto_tune(reader, dummy_snark)
            };
            let halo2_pk = Halo2ProvingKey {
                verifier,
                wrapper,
                profiling: halo2_config.profiling,
            };
            let bytes = bitcode::serialize(&halo2_pk).unwrap();
            std::fs::create_dir_all(halo2_pk_path.parent().unwrap()).unwrap();
            std::fs::write(halo2_pk_path, bytes).unwrap();
            halo2_pk
        } else {
            println!("Loading Halo2 proving key from disk");
            let bytes = std::fs::read(halo2_pk_path).unwrap();
            bitcode::deserialize(&bytes).unwrap()
        };

        Self {
            minimal_stark_pk,
            halo2_pk,
        }
    }
}
