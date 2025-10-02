use crate::arch::aot::AotInstance;
use openvm_instructions::program::Program;
use openvm_instructions::LocalOpcode;
use openvm_instructions::{exe::VmExe, instruction::Instruction};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use p3_baby_bear::BabyBearParameters;
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode;

#[test]
fn test_aot() {
    type F = BabyBear;

    let program = Program::from_instructions(&[
    Instruction {
        opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
        a: F::from_canonical_u32(0),
        b: F::from_canonical_u32(1),
        c: F::from_canonical_u32(0),
        d: F::from_canonical_u32(0),
        e: F::from_canonical_u32(0),
        f: F::from_canonical_u32(0),
        g: F::from_canonical_u32(0),
    }, 
    Instruction {
        opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
        a: F::from_canonical_u32(1),
        b: F::from_canonical_u32(2),
        c: F::from_canonical_u32(0),
        d: F::from_canonical_u32(0),
        e: F::from_canonical_u32(0),
        f: F::from_canonical_u32(0),
        g: F::from_canonical_u32(0),
    }, 
    Instruction {
        opcode: BaseAluOpcode::ADD.global_opcode(),
        a: F::from_canonical_u32(2),
        b: F::from_canonical_u32(0),
        c: F::from_canonical_u32(1),
        d: F::from_canonical_u32(0),
        e: F::from_canonical_u32(0),
        f: F::from_canonical_u32(0),
        g: F::from_canonical_u32(0),
    }]);
    let exe = VmExe {
        program,
        pc_start: 0,
        fn_bounds: Default::default(),
        init_memory: Default::default(),
    };

    let aot_instance = AotInstance::<BabyBear>::new(&exe);
    aot_instance.execute();
}
