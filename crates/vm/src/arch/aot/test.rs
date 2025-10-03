use crate::arch::aot::AotInstance;
use openvm_instructions::program::Program;
use openvm_instructions::LocalOpcode;
use openvm_instructions::{exe::VmExe, instruction::Instruction};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use p3_baby_bear::BabyBearParameters;
use openvm_rv32im_transpiler::{
    Rv32LoadStoreOpcode,
    BranchEqualOpcode,
    BranchLessThanOpcode
};

type F = BabyBear;

fn test_run_program(program: Program<F>) {
    let exe = VmExe {
        program,
        pc_start: 0,
        fn_bounds: Default::default(),
        init_memory: Default::default(),
    };

    let aot_instance = AotInstance::<BabyBear>::new(&exe);
    aot_instance.execute();
}

#[test]
fn test_aot() {
    /*
    test_run_program(
        Program::from_instructions(&[
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
        }])
    );
    */

    /*
    test BEQ
    test_run_program(Program::from_instructions(&[
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
            b: F::from_canonical_u32(1),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        }, 
        Instruction {
            opcode: BranchEqualOpcode::BEQ.global_opcode(),
            a: F::from_canonical_u32(0),
            b: F::from_canonical_u32(1),
            c: F::from_canonical_u32(8),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        },
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(2),
            b: F::from_canonical_u32(2),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        },
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(3),
            b: F::from_canonical_u32(3),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        }])
    );
    */

    // test BGEU
    test_run_program(Program::from_instructions(&[
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(0),
            b: F::from_canonical_u32(0),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        }, 
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(1),
            b: F::from_canonical_u32(1),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        }, 
        Instruction {
            opcode: BranchLessThanOpcode::BGEU.global_opcode(),
            a: F::from_canonical_u32(0),
            b: F::from_canonical_u32(1),
            c: F::from_canonical_u32(8),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        },
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(2),
            b: F::from_canonical_u32(2),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        },
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(3),
            b: F::from_canonical_u32(3),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        }])
    );


}

fn create_add_inst(a: u32, b: u32, c: u32) -> Instruction<F> {
    Instruction {
        opcode: BaseAluOpcode::ADD.global_opcode(),
        a: F::from_canonical_u32(a),
        b: F::from_canonical_u32(b),
        c: F::from_canonical_u32(c),
        d: F::from_canonical_u32(0),
        e: F::from_canonical_u32(0),
        f: F::from_canonical_u32(0),
        g: F::from_canonical_u32(0)
    }
}

fn create_load_inst(a: u32, b: u32, e: u32) -> Instruction<F> {
    Instruction {
        opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
        a: F::from_canonical_u32(a),
        b: F::from_canonical_u32(b),
        c: F::from_canonical_u32(0),
        d: F::from_canonical_u32(0),
        e: F::from_canonical_u32(e),
        f: F::from_canonical_u32(0),
        g: F::from_canonical_u32(0),
    }
}

// TODO: finish creating this in-progress fibonacci test
#[test]
fn test_aot_fibonacci() {
    // 0, 1, 1, 2, 3, 5,
    test_run_program(Program::from_instructions(&[
        create_load_inst(0, 3, 0), // register 0 stores n, we set it to 3 (can change to any)
        create_load_inst(1, 0, 0), // register 1 stores t0, we set it to 0
        create_load_inst(2, 1, 0), // register 2 stores t1, we set it to 1
        create_load_inst(3, 0, 0), // register 3 stores t2, we set it to 0
        create_add_inst(3, 1, 2), // t2 = t0 + t1
        create_load_inst(1, 2, 1), // t0 <- t1
        create_load_inst(2, 3, 1), // t1 <- t2
        create_add_inst(3, 3, 1), // t2 <- t2 + 1
        Instruction::from_isize(
            BranchLessThanOpcode::BGEU.global_opcode(),
            3,
            0,
            0,
            0,
            0
        )
    ]));
}