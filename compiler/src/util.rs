use p3_field::PrimeField32;
use stark_vm::{
    cpu::trace::Instruction,
    vm::{
        config::{VmConfig, VmParamsConfig},
        VirtualMachine,
    },
};

pub fn canonical_i32_to_field<F: PrimeField32>(x: i32) -> F {
    let modulus = F::ORDER_U32;
    assert!(x < modulus as i32 && x >= -(modulus as i32));
    if x < 0 {
        -F::from_canonical_u32((-x) as u32)
    } else {
        F::from_canonical_u32(x as u32)
    }
}

pub fn execute_program<const WORD_SIZE: usize, F: PrimeField32>(program: Vec<Instruction<F>>) {
    let mut vm = VirtualMachine::<WORD_SIZE, _>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                field_extension_enabled: false,
                limb_bits: 28,
                decomp: 4,
            },
        },
        program,
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
