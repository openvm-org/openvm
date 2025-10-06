use crate::arch::aot::AotInstance;
use openvm_instructions::program::Program;
use openvm_instructions::LocalOpcode;
use openvm_instructions::{exe::VmExe, instruction::Instruction};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use p3_baby_bear::{BabyBear, BabyBearParameters};
use crate::arch::VmState;
use crate::arch::MemoryConfig;
use crate::arch::SystemConfig;
use crate::arch::state;

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
    let mut vm_exec_state = aot_instance.execute();
    
    /*
    check if value was actually written correctly
    */

    let value = vm_exec_state.vm_read::<u8, 1>(1, 1);
    println!("value: {:?}", value);

    aot_instance.clean_up();
}

#[test]
fn test_get_vm_state() {
    let program = Program::from_instructions(&[
        Instruction {
            opcode: Rv32LoadStoreOpcode::LOADB.global_opcode(),
            a: F::from_canonical_u32(0),
            b: F::from_canonical_u32(0),
            c: F::from_canonical_u32(0),
            d: F::from_canonical_u32(0),
            e: F::from_canonical_u32(0),
            f: F::from_canonical_u32(0),
            g: F::from_canonical_u32(0),
        }]);

    let exe = VmExe {
        program,
        pc_start: 0,
        fn_bounds: Default::default(),
        init_memory: Default::default()
    };

    let memory_config = MemoryConfig::default();
    let system_config = SystemConfig::default_from_memory(memory_config);
    let init_memory = &exe.init_memory;
    let vm_state: Box<state::VmState<F>> = Box::new(
        VmState::initial(&system_config, &init_memory, 0, vec![])
    );

    let heap_address = &*vm_state as *const _ as usize;
    println!("heap address: 0x{:x}", heap_address);

    // store this heap address on some predetermined location during compile time 
    // this is done to pass in vmstate to the assembly

}

#[test]
fn test_aot() {
    test_run_program(
        Program::from_instructions(&[
            Instruction::from_isize(
                Rv32LoadStoreOpcode::LOADB.global_opcode(),
                0,
                1,
                0,
                0,
                0
            )
        ])
    );
}