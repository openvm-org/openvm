use p3_field::PrimeField32;
use stark_vm::cpu::{
    trace::{Instruction, ProgramExecution},
    CpuAir, CpuOptions,
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

pub fn execute_program<F: PrimeField32>(program: Vec<Instruction<F>>) -> ProgramExecution<F> {
    let cpu = CpuAir::new(CpuOptions {
        field_arithmetic_enabled: true,
    });
    cpu.generate_program_execution(program)
}

pub fn display_program<F: PrimeField32>(program: &[Instruction<F>]) {
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
