use std::sync::{Arc, OnceLock};

#[cfg(feature = "async")]
pub use async_prover::*;
use getset::Getters;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::{vm_poseidon2_hasher, Poseidon2Hasher},
        instructions::exe::VmExe,
        verify_segments, ContinuationVmProof, ContinuationVmProver, Executor, MeteredExecutor,
        PreflightExecutor, VerifiedExecutionPayload, VirtualMachine, VirtualMachineError,
        VmBuilder, VmExecutionConfig, VmInstance, VmVerificationError,
    },
    system::memory::CHUNK,
};
use openvm_stark_backend::{
    config::{Com, Val},
    keygen::types::MultiStarkVerifyingKey,
    p3_field::PrimeField32,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::{StarkEngine, StarkFriEngine},
};
use tracing::instrument;

use crate::{
    commit::{AppExecutionCommit, CommitBytes},
    keygen::AppVerifyingKey,
    prover::vm::{new_local_prover, types::VmProvingKey},
    util::check_max_constraint_degrees,
    StdIn, F, SC,
};

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
    #[getset(get = "pub")]
    leaf_verifier_program_commit: Com<E::SC>,

    app_execution_commit: OnceLock<AppExecutionCommit>,
}

impl<E, VB> AppProver<E, VB>
where
    E: StarkFriEngine,
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
        app_vm_pk: &VmProvingKey<E::SC, VB::VmConfig>,
        app_exe: Arc<VmExe<Val<E::SC>>>,
        leaf_verifier_program_commit: Com<E::SC>,
    ) -> Result<Self, VirtualMachineError> {
        let instance = new_local_prover(vm_builder, app_vm_pk, app_exe)?;
        let app_vm_vk = app_vm_pk.vm_pk.get_vk();

        Ok(Self::new_from_instance(
            instance,
            app_vm_vk,
            leaf_verifier_program_commit,
        ))
    }

    pub fn new_from_instance(
        instance: VmInstance<E, VB>,
        app_vm_vk: MultiStarkVerifyingKey<E::SC>,
        leaf_verifier_program_commit: Com<E::SC>,
    ) -> Self {
        Self {
            program_name: None,
            instance,
            app_vm_vk,
            leaf_verifier_program_commit,
            app_execution_commit: OnceLock::new(),
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

    /// Returns [AppExecutionCommit], which is a commitment to **both** the App VM and the App
    /// VmExe.
    pub fn app_commit(&self) -> AppExecutionCommit {
        *self.app_execution_commit.get_or_init(|| {
            AppExecutionCommit::compute::<E::SC>(
                &self.instance().vm.config().as_ref().memory_config,
                self.instance().exe(),
                self.instance().program_commitment().clone(),
                self.leaf_verifier_program_commit.clone(),
            )
        })
    }

    pub fn app_program_commit(&self) -> Com<E::SC> {
        self.instance().program_commitment().clone()
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
            &self.instance.vm.engine.fri_params(),
        );
        #[cfg(feature = "metrics")]
        metrics::counter!("fri.log_blowup")
            .absolute(self.instance.vm.engine.fri_params().log_blowup as u64);
        ContinuationVmProver::prove(&mut self.instance, input)
    }

    /// Generates proof for every continuation segment
    ///
    /// This function internally calls [verify_segments] to verify the result before returning the
    /// proof.
    ///
    /// **Note**: This function calls [`app_commit`](Self::app_commit), which is computationally
    /// intensive if it is the first time it is called within an `AppProver` instance.
    #[instrument(name = "app_prove_and_verify", skip_all)]
    pub fn prove_and_verify(
        &mut self,
        input: StdIn<Val<E::SC>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError>
    where
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
            + MeteredExecutor<Val<E::SC>>
            + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        let proofs = self.prove(input)?;
        // We skip verification of the user public values proof here because it is directly computed
        // from the merkle tree above
        let res = verify_segments(
            &self.instance.vm.engine,
            &self.app_vm_vk,
            &proofs.per_segment,
        )?;
        let app_exe_commit_u32s = self.app_commit().app_exe_commit.to_u32_digest();
        let exe_commit_u32s = res.exe_commit.map(|x| x.as_canonical_u32());
        if exe_commit_u32s != app_exe_commit_u32s {
            return Err(VmVerificationError::ExeCommitMismatch {
                expected: app_exe_commit_u32s,
                actual: exe_commit_u32s,
            }
            .into());
        }
        Ok(proofs)
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

/// The payload of a verified guest VM execution with user public values extracted and
/// verified.
pub struct VerifiedAppArtifacts {
    /// The Merklelized hash of:
    /// - Program code commitment (commitment of the cached trace)
    /// - Merkle root of the initial memory
    /// - Starting program counter (`pc_start`)
    ///
    /// The Merklelization uses Poseidon2 as a cryptographic hash function (for the leaves)
    /// and a cryptographic compression function (for internal nodes).
    pub app_exe_commit: CommitBytes,
    pub user_public_values: Vec<u8>,
}

/// Verifies the [ContinuationVmProof], which is a collection of STARK proofs as well as
/// additional Merkle proof for user public values.
///
/// This function verifies the STARK proofs and additional conditions to ensure that the
/// `proof` is a valid proof of guest VM execution that terminates successfully (exit code 0)
/// _with respect to_ a commitment to some VM executable.
/// It is the responsibility of the caller to check that the commitment matches the expected
/// VM executable.
pub fn verify_app_proof(
    app_vk: &AppVerifyingKey,
    proof: &ContinuationVmProof<SC>,
) -> Result<VerifiedAppArtifacts, VmVerificationError> {
    static POSEIDON2_HASHER: OnceLock<Poseidon2Hasher<F>> = OnceLock::new();
    let engine = BabyBearPoseidon2Engine::new(app_vk.fri_params);
    let VerifiedExecutionPayload {
        exe_commit,
        final_memory_root,
    } = verify_segments(&engine, &app_vk.vk, &proof.per_segment)?;

    proof.user_public_values.verify(
        POSEIDON2_HASHER.get_or_init(vm_poseidon2_hasher),
        app_vk.memory_dimensions,
        final_memory_root,
    )?;

    let app_exe_commit = CommitBytes::from_u32_digest(&exe_commit.map(|x| x.as_canonical_u32()));
    // The user public values address space has cells have type u8
    let user_public_values = proof
        .user_public_values
        .public_values
        .iter()
        .map(|x| x.as_canonical_u32().try_into().unwrap())
        .collect_vec();
    Ok(VerifiedAppArtifacts {
        app_exe_commit,
        user_public_values,
    })
}

#[cfg(feature = "async")]
mod async_prover {
    use derivative::Derivative;
    use openvm_circuit::{
        arch::ExecutionError, system::memory::merkle::public_values::UserPublicValuesProof,
    };
    use openvm_stark_sdk::config::FriParameters;
    use tokio::{spawn, sync::Semaphore, task::spawn_blocking};
    use tracing::{info_span, instrument, Instrument};

    use super::*;

    /// Thread-safe asynchronous app prover.
    #[derive(Derivative, Getters)]
    #[derivative(Clone)]
    pub struct AsyncAppProver<E, VB>
    where
        E: StarkEngine,
        VB: VmBuilder<E>,
    {
        pub program_name: Option<String>,
        #[getset(get = "pub")]
        vm_builder: VB,
        #[getset(get = "pub")]
        app_vm_pk: Arc<VmProvingKey<E::SC, VB::VmConfig>>,
        app_exe: Arc<VmExe<Val<E::SC>>>,
        #[getset(get = "pub")]
        leaf_verifier_program_commit: Com<E::SC>,

        semaphore: Arc<Semaphore>,
    }

    impl<E, VB> AsyncAppProver<E, VB>
    where
        E: StarkFriEngine + 'static,
        VB: VmBuilder<E> + Clone + Send + Sync + 'static,
        VB::VmConfig: Send + Sync,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
            + MeteredExecutor<Val<E::SC>>
            + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
        Val<E::SC>: PrimeField32,
        Com<E::SC>:
            AsRef<[Val<E::SC>; CHUNK]> + From<[Val<E::SC>; CHUNK]> + Into<[Val<E::SC>; CHUNK]>,
    {
        pub fn new(
            vm_builder: VB,
            app_vm_pk: Arc<VmProvingKey<E::SC, VB::VmConfig>>,
            app_exe: Arc<VmExe<Val<E::SC>>>,
            leaf_verifier_program_commit: Com<E::SC>,
            max_concurrency: usize,
        ) -> Result<Self, VirtualMachineError> {
            Ok(Self {
                program_name: None,
                vm_builder,
                app_vm_pk,
                app_exe,
                leaf_verifier_program_commit,
                semaphore: Arc::new(Semaphore::new(max_concurrency)),
            })
        }

        pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
            self.program_name = Some(program_name.as_ref().to_string());
            self
        }
        pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
            self.set_program_name(program_name);
            self
        }

        /// App Exe
        pub fn exe(&self) -> Arc<VmExe<Val<E::SC>>> {
            self.app_exe.clone()
        }

        /// App VM config
        pub fn vm_config(&self) -> &VB::VmConfig {
            &self.app_vm_pk.vm_config
        }

        pub fn fri_params(&self) -> FriParameters {
            self.app_vm_pk.fri_params
        }

        /// Creates an [AppProver] within a particular thread. The former instance is not
        /// thread-safe and should **not** be moved between threads.
        pub fn local(&self) -> Result<AppProver<E, VB>, VirtualMachineError> {
            AppProver::new(
                self.vm_builder.clone(),
                &self.app_vm_pk,
                self.app_exe.clone(),
                self.leaf_verifier_program_commit.clone(),
            )
        }

        #[instrument(
            name = "app proof",
            skip_all,
            fields(
                group = self.program_name.as_ref().unwrap_or(&"app_proof".to_string())
            )
        )]
        pub async fn prove(
            self,
            input: StdIn<Val<E::SC>>,
        ) -> eyre::Result<ContinuationVmProof<E::SC>> {
            assert!(self.vm_config().as_ref().continuation_enabled);
            check_max_constraint_degrees(self.vm_config().as_ref(), &self.fri_params());
            #[cfg(feature = "metrics")]
            metrics::counter!("fri.log_blowup").absolute(self.fri_params().log_blowup as u64);

            // PERF[jpw]: it is possible to create metered_interpreter without creating vm. The
            // latter is more convenient, but does unnecessary setup (e.g., transfer pk to
            // device). Also, app_commit should be cached.
            let mut local_prover = self.local()?;
            let app_commit = local_prover.app_commit();
            local_prover.instance.reset_state(input.clone());
            let mut state = local_prover.instance.state_mut().take().unwrap();
            let vm = &mut local_prover.instance.vm;
            let metered_ctx = vm.build_metered_ctx(&self.app_exe);
            let metered_interpreter = vm.metered_interpreter(&self.app_exe)?;
            let (segments, _) = metered_interpreter.execute_metered(input, metered_ctx)?;
            drop(metered_interpreter);
            let pure_interpreter = vm.interpreter(&self.app_exe)?;
            let mut tasks = Vec::with_capacity(segments.len());
            let terminal_instret = segments
                .last()
                .map(|s| s.instret_start + s.num_insns)
                .unwrap_or(u64::MAX);
            for (seg_idx, segment) in segments.into_iter().enumerate() {
                tracing::info!(
                    %seg_idx,
                    instret = state.instret(),
                    %segment.instret_start,
                    pc = state.pc(),
                    "Re-executing",
                );
                let num_insns = segment.instret_start.checked_sub(state.instret()).unwrap();
                state = pure_interpreter.execute_from_state(state, Some(num_insns))?;

                let semaphore = self.semaphore.clone();
                let async_worker = self.clone();
                let start_state = state.clone();
                let task = spawn(
                    async move {
                        let _permit = semaphore.acquire().await?;
                        let span = tracing::Span::current();
                        spawn_blocking(move || {
                            let _span = span.enter();
                            info_span!("prove_segment", segment = seg_idx).in_scope(
                                || -> eyre::Result<_> {
                                    // We need a separate span so the metric label includes
                                    // "segment"
                                    // from _segment_span
                                    let _prove_span = info_span!(
                                        "vm_prove",
                                        thread_id = ?std::thread::current().id()
                                    )
                                    .entered();
                                    let mut worker = async_worker.local()?;
                                    let instance = &mut worker.instance;
                                    let vm = &mut instance.vm;
                                    let preflight_interpreter = &mut instance.interpreter;
                                    let (segment_proof, _) = vm.prove(
                                        preflight_interpreter,
                                        start_state,
                                        Some(segment.num_insns),
                                        &segment.trace_heights,
                                    )?;
                                    Ok(segment_proof)
                                },
                            )
                        })
                        .await?
                    }
                    .in_current_span(),
                );
                tasks.push(task);
            }
            // Finish execution to termination
            state = pure_interpreter.execute_from_state(state, None)?;
            if state.instret() != terminal_instret {
                tracing::warn!(
                    "Pure execution terminal instret={}, metered execution terminal instret={}",
                    state.instret(),
                    terminal_instret
                );
                // This should never happen
                return Err(ExecutionError::DidNotTerminate.into());
            }
            let final_memory = &state.memory.memory;
            let user_public_values = UserPublicValuesProof::compute(
                vm.config().as_ref().memory_config.memory_dimensions(),
                vm.config().as_ref().num_public_values,
                &vm_poseidon2_hasher(),
                final_memory,
            );

            let mut proofs = Vec::with_capacity(tasks.len());
            for task in tasks {
                let proof = task.await??;
                proofs.push(proof);
            }
            let cont_proof = ContinuationVmProof {
                per_segment: proofs,
                user_public_values,
            };

            // We skip verification of the user public values proof here because it is directly
            // computed from the merkle tree above
            let engine = E::new(self.fri_params());
            let res = verify_segments(
                &engine,
                &self.app_vm_pk.vm_pk.get_vk(),
                &cont_proof.per_segment,
            )?;
            let app_exe_commit_u32s = app_commit.app_exe_commit.to_u32_digest();
            let exe_commit_u32s = res.exe_commit.map(|x| x.as_canonical_u32());
            if exe_commit_u32s != app_exe_commit_u32s {
                return Err(VmVerificationError::ExeCommitMismatch {
                    expected: app_exe_commit_u32s,
                    actual: exe_commit_u32s,
                }
                .into());
            }
            Ok(cont_proof)
        }
    }
}
