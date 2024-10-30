use std::{
    borrow::Borrow, cell::RefCell, collections::VecDeque, marker::PhantomData, mem, sync::Arc,
};

use ax_stark_backend::{
    config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    p3_commit::PolynomialSpace,
    prover::types::{Proof, ProofInput},
    verifier::VerificationError,
};
use axvm_instructions::exe::AxVmExe;
use p3_field::{AbstractField, PrimeField32};
use parking_lot::Mutex;

use super::{CONNECTOR_AIR_ID, MERKLE_AIR_ID};
use crate::{
    arch::{ExecutionSegment, PersistenceType, VmConfig},
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{memory_image_to_equipartition, merkle::MemoryMerklePvs, CHUNK},
        program::{trace::AxVmCommittedExe, ExecutionError},
    },
};

#[derive(Clone, Default, Debug)]
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
}

impl<F> Streams<F> {
    pub fn new(input_stream: impl Into<VecDeque<Vec<F>>>) -> Self {
        Self {
            input_stream: input_stream.into(),
            hint_stream: VecDeque::default(),
        }
    }
}

pub struct VmExecutor<F: PrimeField32> {
    pub config: VmConfig,
    _marker: PhantomData<F>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
}

pub struct VmExecutorResult<SC: StarkGenericConfig> {
    pub per_segment: Vec<ProofInput<SC>>,
}

impl<F: PrimeField32> VmExecutor<F> {
    /// Create a new VM executor with a given config.
    ///
    /// The VM will start with a single segment, which is created from the initial state.
    pub fn new(config: VmConfig) -> Self {
        Self {
            config,
            _marker: Default::default(),
        }
    }

    fn execute_segments(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<Vec<ExecutionSegment<F>>, ExecutionError> {
        let exe = exe.into();
        let streams = Arc::new(Mutex::new(Streams::new(input)));
        let mut segments = vec![];
        let mut segment = ExecutionSegment::new(
            self.config.clone(),
            exe.program.clone(),
            streams.clone(),
            Some(memory_image_to_equipartition(exe.init_memory)),
        );
        let mut pc = exe.pc_start;

        loop {
            let state = segment.execute_from_pc(pc)?;
            pc = state.pc;

            if state.is_terminated {
                break;
            }

            assert_eq!(
                self.config.memory_config.persistence_type,
                PersistenceType::Persistent,
                "cannot segment in volatile memory mode"
            );

            assert_eq!(
                pc,
                segment.chip_set.connector_chip.boundary_states[1]
                    .unwrap()
                    .pc
            );

            let config = mem::take(&mut segment.config);
            let cycle_tracker = mem::take(&mut segment.cycle_tracker);
            let final_memory = mem::take(&mut segment.final_memory)
                .expect("final memory should be set in continuations segment");

            segments.push(segment);

            segment = ExecutionSegment::new(
                config,
                exe.program.clone(),
                streams.clone(),
                Some(final_memory),
            );
            segment.cycle_tracker = cycle_tracker;
        }
        segments.push(segment);
        tracing::debug!("Number of continuation segments: {}", segments.len());

        Ok(segments)
    }

    pub fn execute(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<(), ExecutionError> {
        #[cfg(test)]
        ax_stark_sdk::config::setup_tracing_with_log_level(tracing::Level::WARN);
        self.execute_segments(exe, input).map(|_| ())
    }

    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segments = self.execute_segments(exe, input)?;

        Ok(VmExecutorResult {
            per_segment: segments
                .into_iter()
                .map(|seg| seg.generate_proof_input(None))
                .collect(),
        })
    }
    pub fn execute_and_generate_with_cached_program<SC: StarkGenericConfig>(
        &self,
        commited_exe: Arc<AxVmCommittedExe<SC>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segments = self.execute_segments(commited_exe.exe.clone(), input)?;

        Ok(VmExecutorResult {
            per_segment: segments
                .into_iter()
                .map(|seg| seg.generate_proof_input(Some(commited_exe.committed_program.clone())))
                .collect(),
        })
    }
}

/// A single segment VM.
pub struct SingleSegmentVmExecutor<F: PrimeField32> {
    pub config: VmConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> SingleSegmentVmExecutor<F> {
    pub fn new(config: VmConfig) -> Self {
        assert_eq!(
            config.memory_config.persistence_type,
            PersistenceType::Volatile,
            "Single segment VM only supports volatile memory"
        );
        Self {
            config,
            _marker: Default::default(),
        }
    }

    /// Executes a program and returns the public values. None means the public value is not set.
    pub fn execute(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: Vec<Vec<F>>,
    ) -> Result<Vec<Option<F>>, ExecutionError> {
        let segment = self.execute_impl(exe.into(), input.into())?;
        let pvs = if let Some(pv_chip) = segment.chip_set.public_values_chip {
            let borrowed_pv_chip = RefCell::borrow(&pv_chip);
            borrowed_pv_chip.core.get_custom_public_values()
        } else {
            vec![]
        };
        Ok(pvs)
    }

    /// Executes a program and returns its proof input.
    pub fn execute_and_generate<SC: StarkGenericConfig>(
        &self,
        commited_exe: Arc<AxVmCommittedExe<SC>>,
        input: Vec<Vec<F>>,
    ) -> Result<ProofInput<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let segment = self.execute_impl(commited_exe.exe.clone(), input.into())?;
        Ok(segment.generate_proof_input(Some(commited_exe.committed_program.clone())))
    }

    fn execute_impl(
        &self,
        exe: AxVmExe<F>,
        input: VecDeque<Vec<F>>,
    ) -> Result<ExecutionSegment<F>, ExecutionError> {
        let pc_start = exe.pc_start;
        let mut segment = ExecutionSegment::new(
            self.config.clone(),
            exe.program.clone(),
            Arc::new(Mutex::new(Streams {
                input_stream: input,
                hint_stream: VecDeque::new(),
            })),
            None,
        );
        segment.execute_from_pc(pc_start)?;
        Ok(segment)
    }
}

pub struct VirtualMachine<F: PrimeField32, E: StarkEngine<SC>, SC: StarkGenericConfig> {
    pub engine: E,
    pub config: VmConfig,
    _marker: PhantomData<(F, SC)>,
}

impl<F: PrimeField32, E: StarkEngine<SC>, SC: StarkGenericConfig> VirtualMachine<F, E, SC> {
    pub fn new(engine: E, config: VmConfig) -> Self {
        Self {
            engine,
            config,
            _marker: PhantomData,
        }
    }

    pub fn keygen(&self) -> MultiStarkProvingKey<SC>
    where
        Val<SC>: PrimeField32,
    {
        self.config.generate_pk(self.engine.keygen_builder())
    }

    pub fn execute(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<(), ExecutionError> {
        let executor = VmExecutor::new(self.config.clone());
        executor.execute(exe, input)
    }

    pub fn execute_and_generate(
        &self,
        exe: impl Into<AxVmExe<F>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let executor = VmExecutor::new(self.config.clone());
        executor.execute_and_generate(exe, input)
    }

    pub fn execute_and_generate_with_cached_program(
        &self,
        committed_exe: Arc<AxVmCommittedExe<SC>>,
        input: impl Into<VecDeque<Vec<F>>>,
    ) -> Result<VmExecutorResult<SC>, ExecutionError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let executor = VmExecutor::new(self.config.clone());
        executor.execute_and_generate_with_cached_program(committed_exe, input)
    }

    pub fn prove_single(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        proof_input: ProofInput<SC>,
    ) -> Proof<SC>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        self.engine.prove(pk, proof_input)
    }

    pub fn prove(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        results: VmExecutorResult<SC>,
    ) -> Vec<Proof<SC>>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        results
            .per_segment
            .into_iter()
            .map(|proof_input| self.engine.prove(pk, proof_input))
            .collect()
    }

    pub fn verify_single(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        self.engine.verify(vk, proof)
    }

    pub fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proofs: Vec<Proof<SC>>,
    ) -> Result<(), VerificationError> {
        let mut prev_final_memory_root = None;
        let mut prev_final_pc = None;

        for (i, proof) in proofs.iter().enumerate() {
            let res = self.engine.verify(vk, proof);
            match res {
                Ok(_) => (),
                Err(e) => return Err(e),
            };

            // Check public values.
            for air_proof_data in proof.per_air.iter() {
                let pvs = &air_proof_data.public_values;
                let air_vk = &vk.per_air[air_proof_data.air_id];

                if air_proof_data.air_id == CONNECTOR_AIR_ID {
                    let pvs: &VmConnectorPvs<_> = pvs.as_slice().borrow();

                    if i != 0 {
                        // Check initial pc matches the previous final pc.
                        assert_eq!(pvs.initial_pc, prev_final_pc.unwrap());
                    } else {
                        // TODO: Fetch initial pc from program
                    }
                    prev_final_pc = Some(pvs.final_pc);

                    let expected_is_terminate = i == proofs.len() - 1;
                    assert_eq!(
                        pvs.is_terminate,
                        Val::<SC>::from_bool(expected_is_terminate)
                    );

                    let expected_exit_code = if expected_is_terminate {
                        ExitCode::Success as u32
                    } else {
                        DEFAULT_SUSPEND_EXIT_CODE
                    };
                    assert_eq!(
                        pvs.exit_code,
                        Val::<SC>::from_canonical_u32(expected_exit_code)
                    );
                } else if air_proof_data.air_id == MERKLE_AIR_ID {
                    let pvs: &MemoryMerklePvs<_, CHUNK> = pvs.as_slice().borrow();

                    // Check that initial root matches the previous final root.
                    if i != 0 {
                        assert_eq!(pvs.initial_root, prev_final_memory_root.unwrap());
                    }
                    prev_final_memory_root = Some(pvs.final_root);
                } else {
                    assert_eq!(pvs.len(), 0);
                    assert_eq!(air_vk.params.num_public_values, 0);
                }
            }
        }
        Ok(())
    }
}
