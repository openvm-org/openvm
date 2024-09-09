use std::ops::Deref;

use ax_sdk::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::{fri_params_fast_testing, fri_params_with_80_bits_of_security},
        setup_tracing,
    },
    engine::StarkEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeField, PrimeField32};
use stark_vm::{
    cpu::trace::Instruction,
    program::Program,
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
            modular_multiplication_enabled: true,
            bigint_limb_size: 8,
            ..Default::default()
        },
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
        vm.segments[0].cpu_chip.borrow_mut().public_values[index] = Some(value);
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

pub fn execute_and_prove_program(
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
    config: VmConfig,
) {
    let vm = VirtualMachine::new(config, program, input_stream);

    let result = vm.execute_and_generate().unwrap();
    assert_eq!(
        result.segment_results.len(),
        1,
        "only proving one segment for now"
    );

    let result = &result.segment_results[0];

    let perm = random_perm();
    // blowup factor 8 for poseidon2 chip
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    let engine = engine_from_perm(perm, result.max_log_degree(), fri_params);

    let airs = result.airs.iter().map(|air| air.deref()).collect();

    setup_tracing();
    engine
        .run_simple_test(airs, result.traces.clone(), result.public_values.clone())
        .expect("Verification failed");
}
