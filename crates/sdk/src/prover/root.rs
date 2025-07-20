use getset::Getters;
use itertools::zip_eq;
use openvm_circuit::arch::{
    GenerationError, PreflightExecutionOutput, SingleSegmentVmProver, Streams, VirtualMachineError,
    VmLocalProver,
};
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
use openvm_native_circuit::{NativeConfig, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::hints::Hintable;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2_root::BabyBearPoseidon2RootEngine, FriParameters},
    engine::StarkEngine,
    openvm_stark_backend::proof::Proof,
};

use crate::{
    keygen::{perm::AirIdPermutation, RootVerifierProvingKey},
    prover::vm::new_local_prover,
    RootSC, F, SC,
};

/// Local prover for a root verifier.
#[derive(Getters)]
pub struct RootVerifierLocalProver {
    inner: VmLocalProver<BabyBearPoseidon2RootEngine, NativeConfig>,
    /// The constant trace heights, ordered by AIR ID (the original ordering from VmConfig).
    #[getset(get = "pub")]
    fixed_air_heights: Vec<u32>,
    air_id_perm: AirIdPermutation,
}

impl RootVerifierLocalProver {
    pub fn new(root_verifier_pk: RootVerifierProvingKey) -> Result<Self, VirtualMachineError> {
        let inner = new_local_prover(
            &root_verifier_pk.vm_pk,
            &root_verifier_pk.root_committed_exe,
        )?;
        let fixed_air_heights = root_verifier_pk.air_heights;
        let air_id_perm = AirIdPermutation::compute(&fixed_air_heights);
        Ok(Self {
            inner,
            fixed_air_heights,
            air_id_perm,
        })
    }
    pub fn vm_config(&self) -> &NativeConfig {
        self.inner.vm.config()
    }
    #[allow(dead_code)]
    pub(crate) fn fri_params(&self) -> &FriParameters {
        &self.inner.vm.engine.fri_params
    }

    pub fn execute_for_air_heights(
        &mut self,
        input: RootVmVerifierInput<SC>,
    ) -> Result<Vec<u32>, VirtualMachineError> {
        let exe = self.inner.exe().clone();
        let vm = &mut self.inner.vm;
        assert!(!vm.config().as_ref().continuation_enabled);
        let input = input.write();
        let state = vm.executor().create_initial_state(&exe, input);
        vm.transport_init_memory_to_device(&state.memory);
        let PreflightExecutionOutput {
            system_records,
            record_arenas,
            ..
        } = vm.execute_preflight(exe, state, None, NATIVE_MAX_TRACE_HEIGHTS)?;
        // Note[jpw]: we could in theory extract trace heights from just preflight execution, but
        // that requires special logic in the chips so we will just generate the traces for now
        let ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
        let air_heights = ctx
            .per_air
            .iter()
            .map(|(_, air_ctx)| air_ctx.main_trace_height() as u32)
            .collect();
        Ok(air_heights)
    }
}

impl SingleSegmentVmProver<RootSC> for RootVerifierLocalProver {
    // @dev: If this implementation is generalized to prover backends not using MatrixRecordArena,
    // then it must be ensured that:
    // - the Native extension chips can ensure that, if the record arenas have
    //   `force_matrix_dimensions()` set, then the record arena capacity heights must equal the
    //   trace matrix heights.
    // - any chips that do not use record arenas (currently system memory chips) have a way to force
    //   trace heights as well. We currently use the fact that all non-system periphery chips have
    //   fixed height (in particular, there is no Poseidon2PeripheryChip).
    fn prove(
        &mut self,
        input: impl Into<Streams<F>>,
        _: &[u32],
    ) -> Result<Proof<RootSC>, VirtualMachineError> {
        assert!(!self.vm_config().as_ref().continuation_enabled);
        // The following is unrolled from SingleSegmentVmProver for VmLocalProver and
        // VirtualMachine::prove to add special logic around ensuring trace heights are fixed and
        // then reordering the trace matrices so the heights are sorted.
        let input = input.into();
        let exe = self.inner.exe().clone();
        let vm = &mut self.inner.vm;
        assert!(!vm.config().as_ref().continuation_enabled);
        let state = vm.executor().create_initial_state(&exe, input);
        vm.transport_init_memory_to_device(&state.memory);

        let trace_heights = &self.fixed_air_heights;
        let PreflightExecutionOutput {
            system_records,
            mut record_arenas,
            ..
        } = vm.execute_preflight(exe, state, None, trace_heights)?;
        // record_arenas are created with capacity specified by trace_heights. we must ensure
        // `generate_proving_ctx` does not resize the trace matrices to make them smaller:
        for ra in &mut record_arenas {
            ra.force_matrix_dimensions();
        }
        vm.override_system_trace_heights(trace_heights);

        let mut ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
        // Sanity check: ensure all generated trace matrices actually match the fixed heights.
        for (air_idx, (fixed_height, (idx, air_ctx))) in
            zip_eq(trace_heights, &ctx.per_air).enumerate()
        {
            let fixed_height = *fixed_height as usize;
            if air_idx != *idx {
                return Err(GenerationError::ForceTraceHeightIncorrect {
                    air_idx,
                    actual: 0,
                    expected: fixed_height,
                }
                .into());
            }
            if fixed_height != air_ctx.main_trace_height() {
                return Err(GenerationError::ForceTraceHeightIncorrect {
                    air_idx,
                    actual: air_ctx.main_trace_height(),
                    expected: fixed_height,
                }
                .into());
            }
        }
        // Reorder the AIRs by heights.
        self.air_id_perm.permute(&mut ctx.per_air);
        for (i, (air_idx, _)) in ctx.per_air.iter_mut().enumerate() {
            *air_idx = i;
        }
        let proof = vm.engine.prove(vm.pk(), ctx);
        Ok(proof)
    }
}
