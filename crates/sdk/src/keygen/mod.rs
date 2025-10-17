use std::sync::Arc;

use derivative::Derivative;
// use dummy::{compute_root_proof_heights, dummy_internal_proof_riscv_app_vm};
use openvm_circuit::{
    arch::{AirInventoryError, SystemConfig, VirtualMachine, VirtualMachineError, VmCircuitConfig},
    system::memory::dimensions::MemoryDimensions,
};
use openvm_continuations::verifier::{
    internal::InternalVmVerifierConfig, leaf::LeafVmVerifierConfig, root::RootVmVerifierConfig,
};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder};
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{
    config::Val,
    engine::StarkEngine,
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
    },
};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};
#[cfg(feature = "evm-prove")]
use {
    openvm_continuations::static_verifier::StaticVerifierPvHandler,
    openvm_native_recursion::halo2::{
        utils::Halo2ParamsReader, verifier::Halo2VerifierProvingKey,
        wrapper::Halo2WrapperProvingKey,
    },
};

#[cfg(feature = "evm-prove")]
use crate::config::Halo2Config;
use crate::{
    commit::VmCommittedExe,
    config::{AggregationConfig, AppConfig},
    keygen::{
        dummy::{compute_root_proof_heights, dummy_internal_proof_riscv_app_vm},
        perm::AirIdPermutation,
    },
    prover::vm::types::VmProvingKey,
    util::check_max_constraint_degrees,
    RootSC, SC,
};

pub mod asm;
pub(crate) mod dummy;
pub mod perm;
#[cfg(feature = "evm-prove")]
pub mod static_verifier;

/// This is lightweight to clone as it contains smart pointers to the proving keys.
#[derive(Clone, Serialize, Deserialize)]
pub struct AppProvingKey<VC> {
    /// The committed executable of the leaf verifier program that verifies proofs of the App VM
    /// circuit. The App VM circuit constraints are statically compiled into this executable.
    pub leaf_committed_exe: Arc<VmCommittedExe<SC>>,
    pub leaf_fri_params: FriParameters,
    pub app_vm_pk: Arc<VmProvingKey<SC, VC>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AppVerifyingKey {
    /// We store the FRI parameters used to generate the proof separately.
    pub fri_params: FriParameters,
    /// STARK backend verifying key
    pub vk: MultiStarkVerifyingKey<SC>,
    pub memory_dimensions: MemoryDimensions,
}

/// The STARK proving keys necessary for aggregation of app proofs into a single aggregate STARK
/// proof.
///
/// This is lightweight to clone as it contains smart pointers to the proving keys.
#[derive(Clone, Serialize, Deserialize)]
pub struct AggProvingKey {
    pub leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    pub internal_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    pub internal_committed_exe: Arc<VmCommittedExe<SC>>,
    pub root_verifier_pk: RootVerifierProvingKey,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AggVerifyingKey {
    pub(super) leaf_fri_params: FriParameters,
    pub(super) leaf_vk: MultiStarkVerifyingKey<SC>,
    /// FRI parameters used to generate the last internal proof.
    pub(super) internal_fri_params: FriParameters,
    pub(super) internal_vk: MultiStarkVerifyingKey<SC>,
    pub(super) internal_verifier_program_commit: Com<SC>,
}

/// Attention: the serialized size of this struct is VERY large, usually >10GB.
///
/// This is lightweight to clone as it contains smart pointers to the proving keys.
#[cfg(feature = "evm-prove")]
#[derive(Clone, Serialize, Deserialize)]
pub struct Halo2ProvingKey {
    /// Static verifier to verify a stark proof of the root verifier.
    pub verifier: Arc<Halo2VerifierProvingKey>,
    /// Wrapper circuit to verify static verifier and reduce the verification costs in the final
    /// proof.
    pub wrapper: Arc<Halo2WrapperProvingKey>,
    /// Whether to collect detailed profiling metrics
    pub profiling: bool,
}

impl<VC> AppProvingKey<VC>
where
    VC: Clone + VmCircuitConfig<SC> + AsRef<SystemConfig>,
{
    pub fn keygen(config: AppConfig<VC>) -> Result<Self, AirInventoryError> {
        let app_engine = BabyBearPoseidon2Engine::new(config.app_fri_params.fri_params);
        let app_vm_pk = {
            let vm_pk = config.app_vm_config.create_airs()?.keygen(&app_engine);
            assert!(
                vm_pk.max_constraint_degree
                    <= config.app_fri_params.fri_params.max_constraint_degree()
            );
            check_max_constraint_degrees(
                config.app_vm_config.as_ref(),
                &config.app_fri_params.fri_params,
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
                app_system_config: config.app_vm_config.as_ref().clone(),
                compiler_options: config.compiler_options,
            }
            .build_program(&app_vm_pk.vm_pk.get_vk());
            Arc::new(VmCommittedExe::commit(
                leaf_program.into(),
                leaf_engine.config().pcs(),
            ))
        };
        Ok(Self {
            leaf_committed_exe,
            leaf_fri_params: config.leaf_fri_params.fri_params,
            app_vm_pk: Arc::new(app_vm_pk),
        })
    }

    pub fn num_public_values(&self) -> usize {
        self.app_vm_pk.vm_config.as_ref().num_public_values
    }

    pub fn get_app_vk(&self) -> AppVerifyingKey {
        AppVerifyingKey {
            fri_params: self.app_vm_pk.fri_params,
            vk: self.app_vm_pk.vm_pk.get_vk(),
            memory_dimensions: self
                .app_vm_pk
                .vm_config
                .as_ref()
                .memory_config
                .memory_dimensions(),
        }
    }

    pub fn leaf_verifier_program_commit(&self) -> Com<SC> {
        self.leaf_committed_exe.get_program_commit()
    }

    pub fn app_fri_params(&self) -> FriParameters {
        self.app_vm_pk.fri_params
    }

    pub fn vm_config(&self) -> &VC {
        &self.app_vm_pk.vm_config
    }

    pub fn app_config(&self) -> AppConfig<VC> {
        AppConfig {
            app_fri_params: self.app_fri_params().into(),
            app_vm_config: self.vm_config().clone(),
            leaf_fri_params: self.leaf_fri_params.into(),
            compiler_options: Default::default(),
        }
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

impl AggProvingKey {
    #[instrument(
        name = "agg_stark_keygen",
        fields(group = "agg_stark_keygen"),
        skip_all
    )]
    pub fn keygen(config: AggregationConfig) -> Result<Self, VirtualMachineError> {
        let (pk, _) = Self::dummy_proof_and_keygen(config)?;
        Ok(pk)
    }

    #[tracing::instrument(level = "info", fields(group = "agg_keygen"), skip_all)]
    pub(crate) fn dummy_proof_and_keygen(
        config: AggregationConfig,
    ) -> Result<(Self, Proof<SC>), VirtualMachineError> {
        let leaf_vm_config = config.leaf_vm_config();
        let internal_vm_config = config.internal_vm_config();
        let root_vm_config = config.root_verifier_vm_config();

        let leaf_engine = BabyBearPoseidon2Engine::new(config.leaf_fri_params);
        let leaf_vm_pk = {
            let (_, vm_pk) = VirtualMachine::new_with_keygen(
                leaf_engine,
                NativeCpuBuilder,
                leaf_vm_config.clone(),
            )?;
            assert!(vm_pk.max_constraint_degree <= config.leaf_fri_params.max_constraint_degree());
            check_max_constraint_degrees(&leaf_vm_config.system, &config.leaf_fri_params);
            Arc::new(VmProvingKey {
                fri_params: config.leaf_fri_params,
                vm_config: leaf_vm_config,
                vm_pk,
            })
        };
        let leaf_vm_vk = leaf_vm_pk.vm_pk.get_vk();
        check_recursive_verifier_size(
            &leaf_vm_vk,
            config.leaf_fri_params,
            config.internal_fri_params.log_blowup,
        );

        let internal_engine = BabyBearPoseidon2Engine::new(config.internal_fri_params);
        let (internal_vm, vm_pk) = VirtualMachine::new_with_keygen(
            internal_engine,
            NativeCpuBuilder,
            internal_vm_config.clone(),
        )?;
        check_max_constraint_degrees(&internal_vm_config.system, &config.internal_fri_params);
        assert!(vm_pk.max_constraint_degree <= config.internal_fri_params.max_constraint_degree());
        let internal_vm_pk = Arc::new(VmProvingKey {
            fri_params: config.internal_fri_params,
            vm_config: internal_vm_config,
            vm_pk,
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
        let internal_committed_exe = Arc::new(VmCommittedExe::commit(
            internal_program.into(),
            internal_vm.engine.config().pcs(),
        ));

        let internal_proof = dummy_internal_proof_riscv_app_vm(
            leaf_vm_pk.clone(),
            internal_vm_pk.clone(),
            internal_committed_exe.clone(),
            config.max_num_user_public_values,
        )?;

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
            let (mut vm, mut vm_pk) = VirtualMachine::new_with_keygen(
                root_engine,
                NativeCpuBuilder,
                root_vm_config.clone(),
            )?;
            let root_committed_exe = Arc::new(VmCommittedExe::commit(
                root_program.into(),
                vm.engine.config().pcs(),
            ));

            assert!(vm_pk.max_constraint_degree <= config.root_fri_params.max_constraint_degree());

            let air_heights =
                compute_root_proof_heights(&mut vm, &root_committed_exe, &internal_proof)?;
            let root_air_perm = AirIdPermutation::compute(&air_heights);
            // ATTENTION: make sure to permute everything in vm_pk that references the original AIR
            // ID ordering:
            root_air_perm.permute(&mut vm_pk.per_air);
            #[cfg(not(feature = "legacy-v1-3-evm-verifier"))]
            for thc in &mut vm_pk.trace_height_constraints {
                root_air_perm.permute(&mut thc.coefficients);
            }

            RootVerifierProvingKey {
                vm_pk: Arc::new(VmProvingKey {
                    fri_params: config.root_fri_params,
                    vm_config: root_vm_config,
                    vm_pk,
                }),
                root_committed_exe,
                air_heights,
            }
        };
        Ok((
            Self {
                leaf_vm_pk,
                internal_vm_pk,
                internal_committed_exe,
                root_verifier_pk,
            },
            internal_proof,
        ))
    }

    pub fn get_agg_vk(&self) -> AggVerifyingKey {
        let leaf_fri_params = self.leaf_vm_pk.fri_params;
        let leaf_vk = self.leaf_vm_pk.vm_pk.get_vk();
        let internal_fri_params = self.internal_vm_pk.fri_params;
        let internal_vk = self.internal_vm_pk.vm_pk.get_vk();
        let internal_verifier_program_commit = self.internal_committed_exe.get_program_commit();
        AggVerifyingKey {
            leaf_fri_params,
            leaf_vk,
            internal_fri_params,
            internal_vk,
            internal_verifier_program_commit,
        }
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
/// - Instead of the AIR order specified by VmConfig. AIRs are ordered by trace heights.
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
    /// The constant trace heights, ordered by AIR ID (the original ordering from VmConfig).
    pub air_heights: Vec<u32>,
}

#[cfg(feature = "evm-prove")]
impl RootVerifierProvingKey {
    pub(crate) fn air_id_permutation(&self) -> AirIdPermutation {
        AirIdPermutation::compute(&self.air_heights)
    }
}

#[cfg(feature = "evm-prove")]
impl Halo2ProvingKey {
    /// Attention:
    /// - This function is very expensive. Usually it requires >64GB memory and takes >10 minutes.
    ///   /// - Please make sure SRS(KZG parameters) is already downloaded.
    #[tracing::instrument(level = "info", fields(group = "halo2_keygen"), skip_all)]
    pub fn keygen(
        halo2_config: Halo2Config,
        reader: &impl Halo2ParamsReader,
        pv_handler: &impl StaticVerifierPvHandler,
        agg_pk: &AggProvingKey,
        dummy_internal_proof: Proof<SC>,
    ) -> Result<Self, VirtualMachineError> {
        let dummy_root_proof = agg_pk
            .root_verifier_pk
            .generate_dummy_root_proof(dummy_internal_proof)?;
        let verifier = agg_pk.root_verifier_pk.keygen_static_verifier(
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
        Ok(Halo2ProvingKey {
            verifier: Arc::new(verifier),
            wrapper: Arc::new(wrapper),
            profiling: halo2_config.profiling,
        })
    }
}

/// For internal use only.
pub fn _leaf_keygen(
    fri_params: FriParameters,
    leaf_vm_config: NativeConfig,
) -> Result<Arc<VmProvingKey<SC, NativeConfig>>, AirInventoryError> {
    let leaf_engine = BabyBearPoseidon2Engine::new(fri_params);
    let leaf_vm_pk = info_span!("keygen", group = "leaf").in_scope(|| {
        leaf_vm_config
            .create_airs()
            .map(|airs| airs.keygen(&leaf_engine))
    })?;
    Ok(Arc::new(VmProvingKey {
        fri_params,
        vm_config: leaf_vm_config,
        vm_pk: leaf_vm_pk,
    }))
}
