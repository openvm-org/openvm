use p3_baby_bear::BabyBear;
use p3_field::{PrimeField, PrimeField32};
#[cfg(feature = "sdk")]
pub use sdk::*;

pub const NUM_LIMBS: usize = 32;
pub const LIMB_SIZE: usize = 8;
pub const TWO_NUM_LIMBS: usize = 2 * NUM_LIMBS;

use stark_vm::{
    arch::ExecutorName,
    program::{Instruction, Program},
    vm::{config::VmConfig, VirtualMachine},
};

pub fn execute_program_with_config(
    config: VmConfig,
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let vm = VirtualMachine::new(config, program, input_stream);
    vm.execute().unwrap();
}

/// Converts a prime field element to a usize.
pub fn prime_field_to_usize<F: PrimeField>(x: F) -> usize {
    let bu = x.as_canonical_biguint();
    let digits = bu.to_u64_digits();
    if digits.is_empty() {
        return 0;
    }
    let ret = digits[0] as usize;
    for i in 1..digits.len() {
        assert_eq!(digits[i], 0, "Prime field element too large");
    }
    ret
}

pub fn execute_program(program: Program<BabyBear>, input_stream: Vec<Vec<BabyBear>>) {
    let vm = VirtualMachine::new(
        VmConfig {
            num_public_values: 4,
            max_segment_len: (1 << 25) - 100,
            bigint_limb_size: 8,
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
    let vm = VirtualMachine::new(
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

pub fn display_program<F: PrimeField32>(program: &[Instruction<F>]) {
    for instruction in program.iter() {
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
            "{:?} {} {} {} {} {} {} {} {}",
            opcode, op_a, op_b, op_c, d, e, op_f, op_g, debug
        );
    }
}

pub fn display_program_with_pc<F: PrimeField32>(program: &[Instruction<F>]) {
    for (pc, instruction) in program.iter().enumerate() {
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
    use stark_vm::{program::Program, sdk::gen_vm_program_stark_for_test, vm::config::VmConfig};

    pub fn execute_and_prove_program<SC: StarkGenericConfig, E: StarkFriEngine<SC>>(
        program: Program<Val<SC>>,
        input_stream: Vec<Vec<Val<SC>>>,
        config: VmConfig,
        engine: &E,
    ) -> Result<(VerificationDataWithFriParams<SC>, Vec<Vec<Val<SC>>>), VerificationError>
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
        let stark_for_test = gen_vm_program_stark_for_test(program, input_stream, config);
        let pvs = stark_for_test
            .air_infos
            .iter()
            .map(|air| air.public_values.clone())
            .collect();
        let vparams = stark_for_test.run_test(engine)?;
        span.exit();
        Ok((vparams, pvs))
    }
}
