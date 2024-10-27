use axvm_instructions::program::Program;
use p3_baby_bear::BabyBear;
#[cfg(feature = "sdk")]
pub use sdk::*;

use crate::arch::{ExecutorName, VirtualMachine, VmConfig};

pub fn execute_program_with_config(
    config: VmConfig,
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let vm = VirtualMachine::new(config).with_input_stream(input_stream);
    vm.execute(program).unwrap();
}

pub fn execute_program(program: Program<BabyBear>, input_stream: Vec<Vec<BabyBear>>) {
    let vm = VirtualMachine::new(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            ..Default::default()
        }
        .add_executor(ExecutorName::Phantom)
        .add_executor(ExecutorName::LoadStore)
        .add_executor(ExecutorName::BranchEqual)
        .add_executor(ExecutorName::Jal)
        .add_executor(ExecutorName::FieldArithmetic)
        .add_executor(ExecutorName::FieldExtension)
        .add_executor(ExecutorName::Poseidon2)
        .add_executor(ExecutorName::ArithmeticLogicUnit256)
        .add_canonical_modulus()
        .add_executor(ExecutorName::Secp256k1AddUnequal)
        .add_executor(ExecutorName::Secp256k1Double),
    )
    .with_input_stream(input_stream);
    vm.execute(program).unwrap();
}

#[cfg(feature = "sdk")]
mod sdk {
    use ax_sdk::{
        ax_stark_backend::{
            config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
            verifier::VerificationError,
        },
        engine::{StarkFriEngine, VerificationDataWithFriParams},
    };
    use axvm_instructions::program::Program;
    use p3_field::PrimeField32;

    use crate::{arch::VmConfig, sdk::gen_vm_program_test_proof_input};

    type ExecuteAndProveResult<SC> =
        Result<(VerificationDataWithFriParams<SC>, Vec<Vec<Val<SC>>>), VerificationError>;

    pub fn execute_and_prove_program<SC: StarkGenericConfig, E: StarkFriEngine<SC>>(
        program: Program<Val<SC>>,
        input_stream: Vec<Vec<Val<SC>>>,
        config: VmConfig,
        engine: &E,
    ) -> ExecuteAndProveResult<SC>
    where
        Val<SC>: PrimeField32,
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let span = tracing::info_span!("execute_and_prove_program").entered();
        let test_proof_input = gen_vm_program_test_proof_input(program, input_stream, config);
        let pvs = test_proof_input
            .per_air
            .iter()
            .map(|air| air.raw.public_values.clone())
            .collect();
        let vparams = test_proof_input.run_test(engine)?;
        span.exit();
        Ok((vparams, pvs))
    }
}
