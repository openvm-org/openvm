use crate::arch::aot::AotInstance;
use openvm_instructions::program::Program;
use openvm_instructions::{exe::VmExe, instruction::Instruction};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use p3_baby_bear::BabyBearParameters;

#[test]
fn test_aot() {
    let program = Program::from_instructions(&[]);
    let exe = VmExe {
        program,
        pc_start: 0,
        fn_bounds: Default::default(),
        init_memory: Default::default(),
    };

    let aot_instance = AotInstance::<BabyBear>::new(&exe);
    aot_instance.execute();
}
