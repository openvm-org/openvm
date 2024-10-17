use std::fmt::{self, Display};

use p3_baby_bear::BabyBear;
#[cfg(feature = "sdk")]
pub use sdk::*;

use crate::{
    arch::ExecutorName,
    system::{
        program::{Instruction, Program},
        vm::{config::VmConfig, VirtualMachine},
    },
};

pub fn execute_program_with_config(
    config: VmConfig,
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let mut vm = VirtualMachine::new(config, program, input_stream);
    vm.execute().unwrap();
}

pub fn execute_program(program: Program<BabyBear>, input_stream: Vec<Vec<BabyBear>>) {
    let mut vm = VirtualMachine::new(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            ..Default::default()
        }
        .add_default_executor(ExecutorName::ArithmeticLogicUnit256)
        .add_canonical_modulus()
        .add_default_executor(ExecutorName::Secp256k1AddUnequal)
        .add_default_executor(ExecutorName::Secp256k1Double),
        program,
        input_stream,
    );
    vm.execute().unwrap();
}

pub fn execute_program_with_public_values(
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
    public_values: &[(usize, BabyBear)],
) {
    let mut vm = VirtualMachine::new(
        VmConfig {
            num_public_values: 4,
            ..Default::default()
        },
        program,
        input_stream,
    );
    for &(index, value) in public_values {
        vm.segments[0].core_chip.borrow_mut().public_values[index] = Some(value);
    }
    vm.execute().unwrap()
}

impl<F: Copy + Display> Display for Program<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instruction in self.instructions().iter() {
            let Instruction {
                opcode,
                op_a,
                op_b,
                op_c,
                d,
                e,
                op_f,
                op_g,
                debug,
            } = instruction;
            write!(
                f,
                "{:?} {} {} {} {} {} {} {} {}",
                opcode, op_a, op_b, op_c, d, e, op_f, op_g, debug,
            )?;
        }
        Ok(())
    }
}

pub fn display_program_with_pc<F: Copy + Display>(program: &Program<F>) {
    for (pc, instruction) in program.instructions().iter().enumerate() {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            op_f,
            op_g,
            debug,
        } = instruction;
        println!(
            "{} | {:?} {} {} {} {} {} {} {} {}",
            pc, opcode, op_a, op_b, op_c, d, e, op_f, op_g, debug
        );
    }
}

#[cfg(feature = "sdk")]
mod sdk {
    use ax_sdk::{
        afs_stark_backend::{
            config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
            verifier::VerificationError,
        },
        engine::{StarkFriEngine, VerificationDataWithFriParams},
    };
    use p3_field::PrimeField32;

    use crate::{
        sdk::gen_vm_program_test_proof_input,
        system::{program::Program, vm::config::VmConfig},
    };

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
