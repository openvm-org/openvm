use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use stark_vm::vm::ExecutionResult;

use afs_test_utils::config::baby_bear_poseidon2::{engine_from_perm, random_perm};
use afs_test_utils::config::fri_params::{
    fri_params_fast_testing, fri_params_with_80_bits_of_security,
};
use afs_test_utils::config::setup_tracing;
use afs_test_utils::engine::StarkEngine;
use stark_vm::{
    cpu::trace::Instruction,
    vm::{config::VmConfig, VirtualMachine},
};

use crate::asm::{AsmBuilder, AsmConfig, Program};
use crate::conversion::CompilerOptions;

pub fn canonical_i32_to_field<F: PrimeField32>(x: i32) -> F {
    let modulus = F::ORDER_U32;
    assert!(x < modulus as i32 && x >= -(modulus as i32));
    if x < 0 {
        -F::from_canonical_u32((-x) as u32)
    } else {
        F::from_canonical_u32(x as u32)
    }
}

pub fn execute_program<const WORD_SIZE: usize>(
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            num_public_values: 4,
            ..Default::default()
        },
        program.isa_instructions,
        input_stream,
    );
    vm.execute().unwrap();
}

pub fn execute_program_with_public_values<const WORD_SIZE: usize>(
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
    public_values: &[(usize, BabyBear)],
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            num_public_values: 4,
            ..Default::default()
        },
        program.isa_instructions,
        input_stream,
    );
    for &(index, value) in public_values {
        vm.segments[0].public_values[index] = Some(value);
    }
    vm.execute().unwrap();
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
            debug,
        } = instruction;
        println!(
            "{:?} {} {} {} {} {} {}",
            opcode, op_a, op_b, op_c, d, e, debug
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
            debug,
        } = instruction;
        println!(
            "{} | {:?} {} {} {} {} {} {}",
            pc, opcode, op_a, op_b, op_c, d, e, debug
        );
    }
}
pub fn end_to_end_test<const WORD_SIZE: usize, EF: ExtensionField<BabyBear> + TwoAdicField>(
    builder: AsmBuilder<BabyBear, EF>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let program = builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
        compile_prints: false,
        enable_cycle_tracker: false,
        field_arithmetic_enabled: true,
        field_extension_enabled: true,
    });
    execute_and_prove_program::<WORD_SIZE>(program, input_stream)
}

pub fn execute_and_prove_program<const WORD_SIZE: usize>(
    program: Program<BabyBear>,
    input_stream: Vec<Vec<BabyBear>>,
) {
    let vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            num_public_values: 4,
            ..Default::default()
        },
        program.isa_instructions,
        input_stream,
    );

    let ExecutionResult {
        max_log_degree,
        nonempty_chips: chips,
        nonempty_traces: traces,
        nonempty_pis: pis,
        ..
    } = vm.execute().unwrap();
    let chips = VirtualMachine::<WORD_SIZE, _>::get_chips(&chips);

    let perm = random_perm();
    // blowup factor 8 for poseidon2 chip
    let fri_params = if matches!(std::env::var("AXIOM_FAST_TEST"), Ok(x) if &x == "1") {
        fri_params_fast_testing()[1]
    } else {
        fri_params_with_80_bits_of_security()[1]
    };
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    setup_tracing();
    engine
        .run_simple_test(chips, traces, pis)
        .expect("Verification failed");
}
