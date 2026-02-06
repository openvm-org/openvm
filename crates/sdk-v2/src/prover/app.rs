use std::sync::{Arc, OnceLock};

use getset::Getters;
use openvm_circuit::{
    arch::{
        ContinuationVmProof, ContinuationVmProver, Executor, MeteredExecutor, PreflightExecutor,
        VerifiedExecutionPayload, VirtualMachine, VirtualMachineError, VmBuilder,
        VmExecutionConfig, VmInstance, VmVerificationError,
        hasher::poseidon2::{Poseidon2Hasher, vm_poseidon2_hasher},
        instructions::exe::VmExe,
        verify_segments,
    },
    system::{
        memory::{CHUNK, dimensions::MemoryDimensions},
        program::trace::VmCommittedExe,
    },
};
use openvm_stark_backend::{
    config::{Com, Val},
    p3_field::PrimeField32,
};
use stark_backend_v2::{
    Digest, StarkWhirEngine, keygen::types::MultiStarkVerifyingKeyV2, prover::ProverBackendV2,
};
use tracing::instrument;

use crate::{
    F, SC, StdIn,
    prover::vm::{new_local_prover, types::VmProvingKey},
    util::check_max_constraint_degrees,
};

#[derive(Getters)]
pub struct AppProver<E, VB>
where
    E: StarkWhirEngine,
    VB: VmBuilder<E>,
{
    pub program_name: Option<String>,
    #[getset(get = "pub")]
    instance: VmInstance<E, VB>,
    #[getset(get = "pub")]
    app_vm_vk: MultiStarkVerifyingKeyV2,
    app_exe_commit: OnceLock<Digest>,
}

impl<E, VB> AppProver<E, VB>
where
    E: StarkWhirEngine,
    VB: VmBuilder<E>,
    Val<E::SC>: PrimeField32,
    Com<E::SC>: AsRef<[Val<E::SC>; CHUNK]> + From<[Val<E::SC>; CHUNK]> + Into<[Val<E::SC>; CHUNK]>,
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
        app_vm_vk: MultiStarkVerifyingKeyV2,
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

    pub fn app_program_commit(&self) -> <E::PB as ProverBackendV2>::Commitment {
        *self.instance().program_commitment()
    }

    /// Returns commitment to the executable
    pub fn app_exe_commit(&self) -> Digest {
        *self.app_exe_commit.get_or_init(|| {
            VmCommittedExe::<SC>::compute_exe_commit(
                &self.app_program_commit().into(),
                self.instance.exe(),
                &self.instance.vm.config().as_ref().memory_config,
            )
            .into()
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

    /// Generates proof for every continuation segment
    #[instrument(
        name = "app_prove",
        skip_all,
        fields(group = self.program_name.as_ref().unwrap_or(&"app_proof".to_string()))
    )]
    pub fn prove(
        &mut self,
        input: StdIn<Val<E::SC>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError>
    where
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
            + MeteredExecutor<Val<E::SC>>
            + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        assert!(self.vm_config().as_ref().continuation_enabled);
        check_max_constraint_degrees(
            self.vm_config().as_ref(),
            self.app_vm_vk.inner.max_constraint_degree(),
        );
        let proof = ContinuationVmProver::prove(&mut self.instance, input)?;
        #[cfg(debug_assertions)]
        let _ = verify_app_proof::<E>(&self.app_vm_vk, self.memory_dimensions(), &proof)?;
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

/// Verifies a ContinuationVmProof and returns the app_exe_commit
pub fn verify_app_proof<E: StarkWhirEngine>(
    vk: &MultiStarkVerifyingKeyV2,
    memory_dimensions: MemoryDimensions,
    proof: &ContinuationVmProof<E::SC>,
) -> Result<Digest, VmVerificationError> {
    static POSEIDON2_HASHER: OnceLock<Poseidon2Hasher<F>> = OnceLock::new();
    let engine = E::new(vk.inner.params.clone());
    let VerifiedExecutionPayload {
        exe_commit,
        final_memory_root,
    } = verify_segments(&engine, vk, &proof.per_segment)?;

    proof.user_public_values.verify(
        POSEIDON2_HASHER.get_or_init(vm_poseidon2_hasher),
        memory_dimensions,
        final_memory_root,
    )?;

    Ok(exe_commit)
}
