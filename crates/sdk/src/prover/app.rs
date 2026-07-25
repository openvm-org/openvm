use std::sync::{Arc, OnceLock};

use getset::Getters;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::{
    execution_mode::MeteredCtx,
    rvr::{cuda::GpuRvrProgram, RvrCheckpointPreflightInstance, RvrMeteredInstance},
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::system::memory::merkle::MerkleTree;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::{vm_poseidon2_hasher, Poseidon2Hasher},
        instructions::exe::VmExe,
        verify_segments, ContinuationVmProof, ContinuationVmProver, Executor, MeteredExecutor,
        PreflightExecutor, VerifiedExecutionPayload, VirtualMachine, VirtualMachineError,
        VmBuilder, VmExecutionConfig, VmInstance, VmVerificationError,
    },
    system::{
        memory::dimensions::MemoryDimensions, program::trace::compute_exe_commit_from_mem_config,
    },
};
use openvm_continuations::CommitBytes;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_sdk_config::SdkVmGpuBuilder;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, p3_field::PrimeField32, prover::ProverBackend,
    StarkEngine, Val,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_verify_stark_host::{vk::VerificationBaseline, VmStarkProof};
use tracing::instrument;

#[cfg(all(feature = "cuda", feature = "rvr"))]
use super::{stark::aggregate_app_proof, AggProver, InternalLayerMetadata};
use crate::{
    keygen::AppVerifyingKey,
    prover::vm::{new_local_prover, types::VmProvingKey},
    util::check_max_constraint_degrees,
    SdkError, StdIn, F, SC,
};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::{DeferralInput, DeferralSetup};

#[derive(Getters)]
pub struct AppProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub program_name: Option<String>,
    #[getset(get = "pub")]
    instance: VmInstance<E, VB>,
    #[getset(get = "pub")]
    app_vm_vk: MultiStarkVerifyingKey<E::SC>,
    app_exe_commit: OnceLock<Digest>,
}

impl<E, VB> AppProver<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    Val<E::SC>: PrimeField32,
{
    /// Creates a new [AppProver] instance. This method will re-commit the `exe` program on device.
    /// If a cached version of the program already exists on device, then directly use the
    /// [`Self::new_from_instance`] constructor.
    ///
    /// The `leaf_verifier_program_commit` is the commitment to the program of the leaf verifier
    /// that verifies the App VM circuit. It can be found in the `AppProvingKey`.
    pub fn new(
        vm_builder: VB,
        app_vm_pk: &VmProvingKey<VB::VmConfig>,
        app_exe: Arc<VmExe<Val<E::SC>>>,
    ) -> Result<Self, VirtualMachineError> {
        let instance = new_local_prover(vm_builder, app_vm_pk, app_exe)?;
        let app_vm_vk = app_vm_pk.vm_pk.get_vk();
        Ok(Self::new_from_instance(instance, app_vm_vk))
    }

    pub fn new_from_instance(
        instance: VmInstance<E, VB>,
        app_vm_vk: MultiStarkVerifyingKey<E::SC>,
    ) -> Self {
        Self {
            program_name: None,
            instance,
            app_vm_vk,
            app_exe_commit: OnceLock::new(),
        }
    }

    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.program_name = Some(program_name.as_ref().to_string());
        self
    }

    pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
        self.set_program_name(program_name);
        self
    }

    pub fn app_program_commit(&self) -> <E::PB as ProverBackend>::Commitment {
        *self.instance().program_commitment()
    }

    /// Returns commitment to the executable
    pub fn app_exe_commit(&self) -> Digest {
        *self.app_exe_commit.get_or_init(|| {
            compute_exe_commit_from_mem_config(
                &self.app_program_commit(),
                self.instance.exe(),
                &self.instance.vm.config().as_ref().memory_config,
            )
        })
    }

    pub fn memory_dimensions(&self) -> MemoryDimensions {
        self.instance
            .vm
            .config()
            .as_ref()
            .memory_config
            .memory_dimensions()
    }

    pub fn num_user_pvs(&self) -> usize {
        self.instance.vm.config().as_ref().num_public_values
    }

    /// Generates proof for every continuation segment
    #[instrument(
        name = "app_prove",
        skip_all,
        fields(group = self.program_name.as_ref().unwrap_or(&"app_proof".to_string()))
    )]
    pub fn prove(&mut self, input: StdIn) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError>
    where
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
            + MeteredExecutor<Val<E::SC>>
            + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        check_max_constraint_degrees(
            self.vm_config().as_ref(),
            self.app_vm_vk.inner.max_constraint_degree(),
        );
        let proof = ContinuationVmProver::prove(&mut self.instance, input)?;
        #[cfg(debug_assertions)]
        let _ = verify_app_proof_inner::<E>(
            &self.app_vm_vk,
            self.memory_dimensions(),
            self.num_user_pvs(),
            &proof,
        )
        .expect("app proof verification failed");
        Ok(proof)
    }

    /// App Exe
    pub fn exe(&self) -> Arc<VmExe<Val<E::SC>>> {
        self.instance.exe().clone()
    }

    /// App VM
    pub fn vm(&self) -> &VirtualMachine<E, VB> {
        &self.instance.vm
    }

    /// App VM config
    pub fn vm_config(&self) -> &VB::VmConfig {
        self.instance.vm.config()
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
pub struct PreflightAppProver<'a> {
    app: AppProver<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    runtime: super::rvr::RvrCheckpointRuntime<'a>,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl<'a> PreflightAppProver<'a> {
    pub(crate) fn new(
        app: AppProver<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
        metered: RvrMeteredInstance<'a>,
        metered_ctx: MeteredCtx,
        checkpoint: RvrCheckpointPreflightInstance<'a>,
        gpu_program: GpuRvrProgram,
    ) -> Self {
        Self {
            app,
            runtime: super::rvr::RvrCheckpointRuntime::new(
                metered,
                metered_ctx,
                checkpoint,
                gpu_program,
            ),
        }
    }

    pub fn vm(&self) -> &VirtualMachine<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder> {
        self.app.vm()
    }

    /// Proves every continuation segment through the prepared compact RVR
    /// checkpoint executor and record-free GPU trace generation.
    ///
    /// The prepared prover remains reusable after successful proofs and
    /// execution errors that occur before RVR trace generation begins. An
    /// error from an active RVR trace-generation session is terminal: lookup
    /// counts are not transactional, so retries fail closed and callers must
    /// prepare a new prover.
    #[instrument(
        name = "app_prove",
        skip_all,
        fields(
            group = self.app.program_name.as_ref().unwrap_or(&"app_proof".to_string()),
            backend = "rvr"
        )
    )]
    pub fn prove(&mut self, input: StdIn) -> Result<ContinuationVmProof<SC>, VirtualMachineError> {
        check_max_constraint_degrees(
            self.app.vm_config().as_ref(),
            self.app.app_vm_vk.inner.max_constraint_degree(),
        );
        let proof = super::rvr::prove(&mut self.app.instance, input, &self.runtime)?;
        #[cfg(debug_assertions)]
        let _ = verify_app_proof_inner::<BabyBearPoseidon2GpuEngine>(
            &self.app.app_vm_vk,
            self.app.memory_dimensions(),
            self.app.num_user_pvs(),
            &proof,
        )
        .expect("app proof verification failed");
        Ok(proof)
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
pub struct PreflightStarkProver<'a> {
    app: PreflightAppProver<'a>,
    agg_prover: Arc<AggProver>,
    deferral_setup: DeferralSetup,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl<'a> PreflightStarkProver<'a> {
    pub(crate) fn new(
        app: PreflightAppProver<'a>,
        agg_prover: Arc<AggProver>,
        deferral_setup: DeferralSetup,
    ) -> Self {
        Self {
            app,
            agg_prover,
            deferral_setup,
        }
    }

    /// Generates the app proof through compiled preflight and then runs
    /// the ordinary recursive aggregation pipeline.
    pub fn prove(
        &mut self,
        input: StdIn,
        def_inputs: &[DeferralInput],
    ) -> eyre::Result<(VmStarkProof, InternalLayerMetadata)> {
        let has_deferrals = self.deferral_setup.hook_commit().is_some();
        let memory_dimensions = self.app.app.memory_dimensions();
        let initial_merkle_tree = if has_deferrals {
            let memory = &self
                .app
                .app
                .instance()
                .state()
                .as_ref()
                .expect("initial state should exist before proving")
                .memory
                .memory;
            Some(MerkleTree::from_memory(
                memory,
                &memory_dimensions,
                &vm_poseidon2_hasher(),
            ))
        } else {
            None
        };

        let app_proof = self.app.prove(input)?;
        let final_merkle_tree = if has_deferrals {
            let memory = &self
                .app
                .app
                .instance()
                .state()
                .as_ref()
                .expect("final state should exist after proving")
                .memory
                .memory;
            Some(MerkleTree::from_memory(
                memory,
                &memory_dimensions,
                &vm_poseidon2_hasher(),
            ))
        } else {
            None
        };

        aggregate_app_proof(
            &self.agg_prover,
            &self.deferral_setup,
            app_proof,
            def_inputs,
            memory_dimensions,
            initial_merkle_tree.as_ref(),
            final_merkle_tree.as_ref(),
        )
    }

    pub fn generate_baseline(&self) -> VerificationBaseline {
        VerificationBaseline {
            app_exe_commit: self.app.app.app_exe_commit(),
            memory_dimensions: self.app.app.memory_dimensions(),
            num_user_pvs: self.app.app.num_user_pvs(),
            app_vk_commit: self.agg_prover.leaf_prover.get_vk_commit(false),
            leaf_vk_commit: self
                .agg_prover
                .internal_for_leaf_prover
                .get_vk_commit(false),
            internal_for_leaf_vk_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_vk_commit(false),
            internal_recursive_vk_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_vk_commit(true),
            expected_def_hook_commit: self.deferral_setup.hook_commit(),
        }
    }
}

/// Verifies a ContinuationVmProof and returns the app_exe_commit
pub fn verify_app_proof<E: StarkEngine<SC = SC>>(
    app_vk: &AppVerifyingKey,
    proof: &ContinuationVmProof<E::SC>,
) -> Result<Digest, SdkError> {
    verify_app_proof_inner::<E>(
        &app_vk.vk,
        app_vk.memory_dimensions,
        app_vk.num_user_pvs,
        proof,
    )
}

/// Verifies a ContinuationVmProof and optionally checks that the recovered app_exe_commit matches
/// the expected value.
pub fn verify_app_proof_with_expected_exe_commit<E: StarkEngine<SC = SC>>(
    app_vk: &AppVerifyingKey,
    proof: &ContinuationVmProof<E::SC>,
    expected_exe_commit: Option<Digest>,
) -> Result<(), SdkError> {
    let exe_commit = verify_app_proof::<E>(app_vk, proof)?;
    if let Some(expected_exe_commit) = expected_exe_commit {
        if exe_commit != expected_exe_commit {
            return Err(SdkError::Other(eyre::eyre!(
                "app proof exe commit mismatch: expected {}, actual {}",
                CommitBytes::from(expected_exe_commit),
                CommitBytes::from(exe_commit)
            )));
        }
    }
    Ok(())
}

/// Verifies a ContinuationVmProof from the borrowed components of an
/// [`AppVerifyingKey`], returning the app_exe_commit.
fn verify_app_proof_inner<E: StarkEngine<SC = SC>>(
    vk: &MultiStarkVerifyingKey<SC>,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
    proof: &ContinuationVmProof<E::SC>,
) -> Result<Digest, SdkError> {
    static POSEIDON2_HASHER: OnceLock<Poseidon2Hasher<F>> = OnceLock::new();
    let engine = E::new(vk.inner.params.clone());
    let VerifiedExecutionPayload {
        exe_commit,
        final_memory_root,
    } = verify_segments(&engine, vk, &proof.per_segment)?;

    if proof.user_public_values.public_values.len() != num_user_pvs {
        return Err(SdkError::Other(eyre::eyre!(
            "wrong number of user public values (expected: {}, actual: {})",
            num_user_pvs,
            proof.user_public_values.public_values.len()
        )));
    }

    proof
        .user_public_values
        .verify(
            POSEIDON2_HASHER.get_or_init(vm_poseidon2_hasher),
            memory_dimensions,
            final_memory_root,
        )
        .map_err(VmVerificationError::from)?;

    Ok(exe_commit)
}
