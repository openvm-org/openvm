use openvm_circuit::{
    arch::VirtualMachine,
    utils::{test_cpu_engine, test_system_config},
};
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode,
};
use openvm_keccak256_transpiler::XorinOpcode;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::{Keccak256Config, Keccak256CpuBuilder};

fn reg(index: usize) -> usize {
    index * REGISTER_NUM_LIMBS
}

#[test]
fn xorin_metering_counts_runtime_replay_words() {
    let instructions = [
        Instruction::<BabyBear>::from_usize(
            XorinOpcode::XORIN.global_opcode(),
            [
                reg(1),
                reg(2),
                reg(3),
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let mut init_memory = SparseMemoryImage::default();
    for (register, value) in [(1, 64u64), (2, 256), (3, 136)] {
        init_memory.extend(
            value
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    let exe = VmExe::new(Program::from_instructions(&instructions)).with_init_memory(init_memory);
    let config = Keccak256Config {
        system: test_system_config(),
        ..Default::default()
    };
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256CpuBuilder, config).unwrap();
    let metered_ctx = vm.build_metered_ctx(&exe);
    let (segments, _) = vm
        .metered_instance(&exe)
        .unwrap()
        .execute_metered(Vec::<Vec<u8>>::new(), metered_ctx)
        .unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].num_insns, 2);
    assert_eq!(segments[0].num_preflight_replay_values, 17);
}
