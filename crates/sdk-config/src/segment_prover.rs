use openvm_circuit::{
    arch::{
        execution_mode::Segment, instructions::program::Program,
        interpreter_preflight::PreflightInterpretedInstance, VirtualMachineError,
        VmExecutionConfig, VmInstance, VmState,
    },
    system::memory::online::GuestMemory,
};
use openvm_stark_backend::{proof::Proof, Val};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;

use crate::{SdkVmConfig, SdkVmCpuBuilder, SC};

type InterpretedPreflight =
    PreflightInterpretedInstance<Val<SC>, <SdkVmConfig as VmExecutionConfig<Val<SC>>>::Executor>;

/// Fixed-program prover for standalone, independently scheduled segments.
pub struct SegmentProver {
    preflight: InterpretedPreflight,
    program: Program<Val<SC>>,
}

impl SegmentProver {
    pub fn new(
        instance: &VmInstance<BabyBearPoseidon2CpuEngine, SdkVmCpuBuilder>,
    ) -> Result<Self, VirtualMachineError> {
        let preflight = instance.vm.preflight_interpreter(instance.exe())?;
        let program = instance.exe().program.clone();
        Ok(Self { preflight, program })
    }

    /// Proves one segment from an arbitrary segment-start state.
    ///
    /// Final memory is returned only when the segment terminates successfully.
    pub fn prove(
        &self,
        instance: &mut VmInstance<BabyBearPoseidon2CpuEngine, SdkVmCpuBuilder>,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<(Proof<SC>, Option<GuestMemory>), VirtualMachineError> {
        instance
            .vm
            .prove_segment(&self.preflight, &self.program, state, segment.num_insns)
    }
}
