use p3_field::PrimeField32;
use stark_vm::{
    cpu::trace::Instruction,
    vm::{
        config::{VmConfig, VmParamsConfig},
        VirtualMachine,
    },
};

pub fn execute_program<const WORD_SIZE: usize, F: PrimeField32>(
    program: Vec<Instruction<F>>,
    witness_stream: Vec<Vec<F>>,
) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                field_extension_enabled: false,
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
