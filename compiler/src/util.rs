use p3_field::PrimeField32;
use stark_vm::cpu::{trace::Instruction, CpuChip};

pub fn canonical_i32_to_field<F: PrimeField32>(x: i32) -> F {
    let modulus = F::ORDER_U32;
    assert!(x < modulus as i32 && x >= -(modulus as i32));
    if x < 0 {
        -F::from_canonical_u32((-x) as u32)
    } else {
        F::from_canonical_u32(x as u32)
    }
}

pub fn execute_program<F: PrimeField32>(program: Vec<Instruction<F>>) {
    let cpu = CpuChip::new(true);
    cpu.generate_program_execution(program);
}

pub fn display_program<F: PrimeField32>(program: &Vec<Instruction<F>>) {
    for instruction in program {
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