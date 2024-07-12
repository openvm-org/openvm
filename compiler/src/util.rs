use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};

use afs_test_utils::config::baby_bear_poseidon2::{engine_from_perm, random_perm};
use afs_test_utils::config::fri_params::fri_params_with_80_bits_of_security;
use afs_test_utils::engine::StarkEngine;
use stark_vm::vm::get_chips;
use stark_vm::{
    cpu::trace::Instruction,
    vm::{
        config::{VmConfig, VmParamsConfig},
        VirtualMachine,
    },
};

use crate::asm::AsmBuilder;
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

pub fn execute_program<const WORD_SIZE: usize, F: PrimeField32>(
    program: Vec<Instruction<F>>,
    witness_stream: Vec<Vec<F>>,
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                field_extension_enabled: true,
                limb_bits: 28,
                decomp: 4,
                compress_poseidon2_enabled: true,
                perm_poseidon2_enabled: true,
            },
        },
        program,
        witness_stream,
    );
    vm.traces().unwrap();
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
        } = instruction;
        println!("{:?} {} {} {} {} {}", opcode, op_a, op_b, op_c, d, e);
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
        } = instruction;
        println!(
            "{} | {:?} {} {} {} {} {}",
            pc, opcode, op_a, op_b, op_c, d, e
        );
    }
}
pub fn end_to_end_test<const WORD_SIZE: usize, EF: ExtensionField<BabyBear> + TwoAdicField>(
    builder: AsmBuilder<BabyBear, EF>,
    witness_stream: Vec<Vec<BabyBear>>,
) {
    let program = builder.compile_isa_with_options::<WORD_SIZE>(CompilerOptions {
        compile_prints: false,
        field_arithmetic_enabled: true,
        field_extension_enabled: true,
    });
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                field_extension_enabled: true,
                limb_bits: 28,
                decomp: 4,
                compress_poseidon2_enabled: true,
                perm_poseidon2_enabled: true,
            },
        },
        program,
        witness_stream,
    );
    let max_log_degree = vm.max_log_degree().unwrap();
    let traces = vm.traces().unwrap();
    let chips = get_chips(&vm);

    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    let num_chips = chips.len();

    engine
        .run_simple_test(chips, traces, vec![vec![]; num_chips])
        .expect("Verification failed");
}
