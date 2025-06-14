use std::sync::Arc;

use derivative::Derivative;
use dummy::{compute_root_proof_heights, dummy_internal_proof_riscv_app_vm};
use openvm_circuit::{
    arch::{VirtualMachine, VmComplexTraceHeights, VmConfig},
    system::{memory::dimensions::MemoryDimensions, program::trace::VmCommittedExe},
};
use openvm_continuations::verifier::{
    internal::InternalVmVerifierConfig, leaf::LeafVmVerifierConfig, root::RootVmVerifierConfig,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{
    config::Val,
    p3_field::{FieldExtensionAlgebra, PrimeField32, TwoAdicField},
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
        proof::Proof,
        Chip,
    },
    p3_bn254_fr::Bn254Fr,
};
use serde::{Deserialize, Serialize};
use tracing::info_span;
#[cfg(feature = "evm-prove")]
use {
    crate::config::AggConfig,
    openvm_continuations::static_verifier::StaticVerifierPvHandler,
    openvm_native_recursion::halo2::{
        utils::Halo2ParamsReader, verifier::Halo2VerifierProvingKey,
        wrapper::Halo2WrapperProvingKey,
    },
};

use crate::{
    commit::babybear_digest_to_bn254,
    config::{AggStarkConfig, AppConfig},
    keygen::perm::AirIdPermutation,
    prover::vm::types::VmProvingKey,
    NonRootCommittedExe, RootSC, F, SC,
};

pub mod asm;
pub(crate) mod dummy;
pub mod perm;
#[cfg(feature = "evm-prove")]
pub mod static_verifier;

#[derive(Clone, Serialize, Deserialize)]
pub struct AppProvingKey<VC> {
    pub leaf_committed_exe: Arc<NonRootCommittedExe>,
    pub leaf_fri_params: FriParameters,
    pub app_vm_pk: Arc<VmProvingKey<SC, VC>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AppVerifyingKey {
    pub fri_params: FriParameters,
    pub app_vm_vk: MultiStarkVerifyingKey<SC>,
    pub memory_dimensions: MemoryDimensions,
}

#[cfg(feature = "evm-prove")]
#[derive(Clone, Serialize, Deserialize)]
pub struct AggProvingKey {
    pub agg_stark_pk: AggStarkProvingKey,
    pub halo2_pk: Halo2ProvingKey,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AggStarkProvingKey {
    pub leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    pub internal_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    pub internal_committed_exe: Arc<NonRootCommittedExe>,
    pub root_verifier_pk: RootVerifierProvingKey,
}

/// Attention: the size of this struct is VERY large, usually >10GB.
#[cfg(feature = "evm-prove")]
#[derive(Clone, Serialize, Deserialize)]
pub struct Halo2ProvingKey {
    /// Static verifier to verify a stark proof of the root verifier.
    pub verifier: Halo2VerifierProvingKey,
    /// Wrapper circuit to verify static verifier and reduce the verification costs in the final
    /// proof.
    pub wrapper: Halo2WrapperProvingKey,
    /// Whether to collect detailed profiling metrics
    pub profiling: bool,
}

impl<VC: VmConfig<F>> AppProvingKey<VC>
where
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn keygen(config: AppConfig<VC>) -> Self {
        let app_engine = BabyBearPoseidon2Engine::new(config.app_fri_params.fri_params);
        let app_vm_pk = {
            let vm = VirtualMachine::new(app_engine, config.app_vm_config.clone());
            let vm_pk = vm.keygen();
            assert!(
                vm_pk.max_constraint_degree
                    <= config.app_fri_params.fri_params.max_constraint_degree()
            );
            VmProvingKey {
                fri_params: config.app_fri_params.fri_params,
                vm_config: config.app_vm_config.clone(),
                vm_pk,
            }
        };
        check_recursive_verifier_size(
            &app_vm_pk.vm_pk.get_vk(),
            config.app_fri_params.fri_params,
            config.leaf_fri_params.fri_params.log_blowup,
        );
        let leaf_committed_exe = {
            let leaf_engine = BabyBearPoseidon2Engine::new(config.leaf_fri_params.fri_params);
            let leaf_program = LeafVmVerifierConfig {
                app_fri_params: config.app_fri_params.fri_params,
                app_system_config: config.app_vm_config.system().clone(),
                compiler_options: config.compiler_options,
            }
            .build_program(&app_vm_pk.vm_pk.get_vk());
            Arc::new(VmCommittedExe::commit(
                leaf_program.into(),
                leaf_engine.config.pcs(),
            ))
        };
        Self {
            leaf_committed_exe,
            leaf_fri_params: config.leaf_fri_params.fri_params,
            app_vm_pk: Arc::new(app_vm_pk),
        }
    }

    pub fn num_public_values(&self) -> usize {
        self.app_vm_pk.vm_config.system().num_public_values
    }

    pub fn get_app_vk(&self) -> AppVerifyingKey {
        AppVerifyingKey {
            fri_params: self.app_vm_pk.fri_params,
            app_vm_vk: self.app_vm_pk.vm_pk.get_vk(),
            memory_dimensions: self
                .app_vm_pk
                .vm_config
                .system()
                .memory_config
                .memory_dimensions(),
        }
    }

    pub fn app_fri_params(&self) -> FriParameters {
        self.app_vm_pk.fri_params
    }

    pub fn commit_in_bn254(&self) -> Bn254Fr {
        babybear_digest_to_bn254(&self.commit_in_babybear())
    }

    pub fn commit_in_babybear(&self) -> [F; DIGEST_SIZE] {
        self.leaf_committed_exe.get_program_commit().into()
    }
}

/// Try to determine statically if there will be an issue with the recursive verifier size and log
/// a warning if so.
///
/// `next_log_blowup` refers to the `log_blowup` of the next verifier in the chain; this determines
/// a maximum trace height.
fn check_recursive_verifier_size<SC: StarkGenericConfig>(
    vk: &MultiStarkVerifyingKey<SC>,
    fri_params: FriParameters,
    next_log_blowup: usize,
) where
    Val<SC>: PrimeField32 + TwoAdicField,
{
    let vk = &vk.inner;

    // for each round we will compute the pair (total_width, num_airs, num_pts)
    let mut rounds = vec![];

    // Preprocessed rounds.
    rounds.extend(
        vk.per_air
            .iter()
            .filter_map(|vk| vk.params.width.preprocessed)
            .map(|width| (width, 1, 2)),
    );

    let common_main_total_width = vk
        .per_air
        .iter()
        .map(|vk| vk.params.width.common_main)
        .sum();
    rounds.push((common_main_total_width, vk.per_air.len(), 2));

    for vk in vk.per_air.iter() {
        for &cached_main_width in &vk.params.width.cached_mains {
            rounds.push((cached_main_width, 1, 2));
        }
    }

    let mut after_challenge_rounds = vec![];
    for vk in vk.per_air.iter() {
        let widths = &vk.params.width.after_challenge;
        if widths.len() > after_challenge_rounds.len() {
            after_challenge_rounds.resize(widths.len(), (0, 0, 2));
        }
        for (i, &width) in widths.iter().enumerate() {
            after_challenge_rounds[i].0 += SC::Challenge::D * width;
            after_challenge_rounds[i].1 += 1;
        }
    }
    rounds.extend(after_challenge_rounds);

    let quotient_round = (
        vk.per_air
            .iter()
            .map(|vk| SC::Challenge::D * vk.quotient_degree as usize)
            .sum(),
        vk.per_air.len(),
        1,
    );
    rounds.push(quotient_round);

    // This computes the number of rows in the `FRI_REDUCED_OPENING` chip, which is the expected
    // bottleneck of the recursive verifier.
    let fri_reduced_opening_trace_height = fri_params.num_queries
        * rounds
            .iter()
            .map(|(total_width, num_airs, total_pts)| total_pts * (total_width + 2 * num_airs))
            .sum::<usize>();
    // First check: is FriReducedOpening trace height too large?
    if fri_reduced_opening_trace_height > (1 << (Val::<SC>::TWO_ADICITY - next_log_blowup)) {
        tracing::warn!("recursive verifier size may be too large; FriReducedOpening height ({fri_reduced_opening_trace_height}) > {}", 1 << (Val::<SC>::TWO_ADICITY - next_log_blowup));
    }
    // Second check: static check for log up soundness constraints using FriReducedOpening trace
    // height as proxy
    if fri_reduced_opening_trace_height as u32 >= Val::<SC>::ORDER_U32 / 200 {
        tracing::warn!(
            "recursive verifier size may violate log up soundness constraints; {} > {}",
            200 * fri_reduced_opening_trace_height,
            Val::<SC>::ORDER_U32
        );
    }
}

impl AggStarkProvingKey {
    pub fn keygen(config: AggStarkConfig) -> Self {
        tracing::info_span!("agg_stark_keygen", group = "agg_stark_keygen")
            .in_scope(|| Self::dummy_proof_and_keygen(config).0)
    }

    pub fn dummy_proof_and_keygen(config: AggStarkConfig) -> (Self, Proof<SC>) {
        let leaf_vm_config = config.leaf_vm_config();
        let internal_vm_config = config.internal_vm_config();
        let root_vm_config = config.root_verifier_vm_config();

        let leaf_engine = BabyBearPoseidon2Engine::new(config.leaf_fri_params);
        let leaf_vm_pk = Arc::new({
            let vm = VirtualMachine::new(leaf_engine, leaf_vm_config.clone());
            let vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.leaf_fri_params.max_constraint_degree());
            VmProvingKey {
                fri_params: config.leaf_fri_params,
                vm_config: leaf_vm_config,
                vm_pk,
            }
        });
        let leaf_vm_vk = leaf_vm_pk.vm_pk.get_vk();
        check_recursive_verifier_size(
            &leaf_vm_vk,
            config.leaf_fri_params,
            config.internal_fri_params.log_blowup,
        );

        let internal_engine = BabyBearPoseidon2Engine::new(config.internal_fri_params);
        let internal_vm = VirtualMachine::new(internal_engine, internal_vm_config.clone());
        let internal_vm_pk = Arc::new({
            let vm_pk = internal_vm.keygen();
            assert!(
                vm_pk.max_constraint_degree <= config.internal_fri_params.max_constraint_degree()
            );
            VmProvingKey {
                fri_params: config.internal_fri_params,
                vm_config: internal_vm_config,
                vm_pk,
            }
        });
        let internal_vm_vk = internal_vm_pk.vm_pk.get_vk();
        check_recursive_verifier_size(
            &internal_vm_vk,
            config.internal_fri_params,
            config.internal_fri_params.log_blowup,
        );

        let internal_program = InternalVmVerifierConfig {
            leaf_fri_params: config.leaf_fri_params,
            internal_fri_params: config.internal_fri_params,
            compiler_options: config.compiler_options,
        }
        .build_program(&leaf_vm_vk, &internal_vm_vk);
        let internal_committed_exe = Arc::new(VmCommittedExe::<SC>::commit(
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
            let mut root_engine = BabyBearPoseidon2RootEngine::new(config.root_fri_params);
            root_engine.max_constraint_degree = config.root_max_constraint_degree;
            let root_program = RootVmVerifierConfig {
                leaf_fri_params: config.leaf_fri_params,
                internal_fri_params: config.internal_fri_params,
                num_user_public_values: config.max_num_user_public_values,
                internal_vm_verifier_commit: internal_committed_exe.get_program_commit().into(),
                compiler_options: config.compiler_options,
            }
            .build_program(&leaf_vm_vk, &internal_vm_vk);
            let root_committed_exe = Arc::new(VmCommittedExe::<RootSC>::commit(
                root_program.into(),
                root_engine.config.pcs(),
            ));

            let vm = VirtualMachine::new(root_engine, root_vm_config.clone());
            let mut vm_pk = vm.keygen();
            assert!(vm_pk.max_constraint_degree <= config.root_fri_params.max_constraint_degree());

            let (air_heights, vm_heights) = compute_root_proof_heights(
                root_vm_config.clone(),
                root_committed_exe.exe.clone(),
                &internal_proof,
            );
            let root_air_perm = AirIdPermutation::compute(&air_heights);
            root_air_perm.permute(&mut vm_pk.per_air);

            RootVerifierProvingKey {
                vm_pk: Arc::new(VmProvingKey {
                    fri_params: config.root_fri_params,
                    vm_config: root_vm_config,
                    vm_pk,
                }),
                root_committed_exe,
                air_heights,
                vm_heights,
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

    pub fn num_user_public_values(&self) -> usize {
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
/// - Instead of the AIR order specified by VC. AIRs are ordered by trace heights.
#[derive(Serialize, Deserialize, Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct RootVerifierProvingKey {
    /// VM Proving key for the root verifier.
    /// - AIR proving key in `MultiStarkProvingKey` is ordered by trace height.
    /// - `VmConfig.overridden_executor_heights` is specified and is in the original AIR order.
    /// - `VmConfig.memory_config.boundary_air_height` is specified.
    pub vm_pk: Arc<VmProvingKey<RootSC, NativeConfig>>,
    /// Committed executable for the root VM.
    pub root_committed_exe: Arc<VmCommittedExe<RootSC>>,
    /// The constant trace heights, ordered by AIR ID.
    pub air_heights: Vec<usize>,
    /// The constant trace heights in a semantic way for VM.
    pub vm_heights: VmComplexTraceHeights,
}

impl RootVerifierProvingKey {
    pub fn air_id_permutation(&self) -> AirIdPermutation {
        AirIdPermutation::compute(&self.air_heights)
    }
}

#[cfg(feature = "evm-prove")]
impl AggProvingKey {
    /// Attention:
    /// - This function is very expensive. Usually it requires >64GB memory and takes >10 minutes.
    /// - Please make sure SRS(KZG parameters) is already downloaded.
    #[tracing::instrument(level = "info", fields(group = "agg_keygen"), skip_all)]
    pub fn keygen(
        config: AggConfig,
        reader: &impl Halo2ParamsReader,
        pv_handler: &impl StaticVerifierPvHandler,
    ) -> Self {
        let AggConfig {
            agg_stark_config,
            halo2_config,
        } = config;
        let (agg_stark_pk, dummy_internal_proof) =
            AggStarkProvingKey::dummy_proof_and_keygen(agg_stark_config);
        let dummy_root_proof = agg_stark_pk
            .root_verifier_pk
            .generate_dummy_root_proof(dummy_internal_proof);
        let verifier = agg_stark_pk.root_verifier_pk.keygen_static_verifier(
            &reader.read_params(halo2_config.verifier_k),
            dummy_root_proof,
            pv_handler,
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
        Self {
            agg_stark_pk,
            halo2_pk,
        }
    }
}

pub fn leaf_keygen(
    fri_params: FriParameters,
    leaf_vm_config: NativeConfig,
) -> Arc<VmProvingKey<SC, NativeConfig>> {
    let leaf_engine = BabyBearPoseidon2Engine::new(fri_params);
    let leaf_vm_pk = info_span!("keygen", group = "leaf")
        .in_scope(|| VirtualMachine::new(leaf_engine, leaf_vm_config.clone()).keygen());
    Arc::new(VmProvingKey {
        fri_params,
        vm_config: leaf_vm_config,
        vm_pk: leaf_vm_pk,
    })
}
