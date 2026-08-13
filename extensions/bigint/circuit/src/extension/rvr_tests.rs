use openvm_bigint_transpiler::{BaseAlu256Opcode, BranchEqual256Opcode};
use openvm_circuit::{
    arch::{rvr::PreflightLimits, VmExecutor},
    utils::test_system_config,
};
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_BYTES},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{BaseAluOpcode, BranchEqualOpcode};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::Int256Rv64Config;

const DST_PTR: u32 = 0x100;
const LHS_PTR: u32 = 0x200;
const RHS_PTR: u32 = 0x300;

fn reg(index: usize) -> usize {
    index * REGISTER_BYTES as usize
}

fn operands(equal: bool) -> ([u8; 32], [u8; 32]) {
    let lhs =
        std::array::from_fn::<_, 32, _>(|index| (index as u8).wrapping_mul(17).wrapping_add(3));
    let rhs = if equal {
        lhs
    } else {
        let mut rhs = lhs;
        rhs[31] ^= 1;
        rhs
    };
    (lhs, rhs)
}

fn add_replay_values(equal: bool) -> [u64; 4] {
    let (lhs, rhs) = operands(equal);
    let mut carry = false;
    std::array::from_fn(|index| {
        let lhs = u64::from_le_bytes(lhs[index * 8..index * 8 + 8].try_into().unwrap());
        let rhs = u64::from_le_bytes(rhs[index * 8..index * 8 + 8].try_into().unwrap());
        let (sum, carry1) = lhs.overflowing_add(rhs);
        let (sum, carry2) = sum.overflowing_add(u64::from(carry));
        carry = carry1 || carry2;
        sum
    })
}

fn fixture_with_pointer_offset(equal: bool, pointer_offset: u32) -> VmExe<BabyBear> {
    let program = Program::from_instructions(&[
        Instruction::from_usize(
            BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode(),
            [
                reg(1),
                reg(2),
                reg(3),
                REGISTER_AS as usize,
                MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(
            BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode(),
            [reg(2), reg(3), 8, REGISTER_AS as usize, MEMORY_AS as usize],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ]);
    let (lhs, rhs) = operands(equal);
    let mut memory = SparseMemoryImage::default();
    for (register, pointer) in [(1, DST_PTR), (2, LHS_PTR), (3, RHS_PTR)] {
        let pointer = pointer + pointer_offset;
        memory.extend(
            u64::from(pointer)
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    memory.extend(
        lhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, LHS_PTR + pointer_offset + offset as u32), byte)),
    );
    memory.extend(
        rhs.into_iter()
            .enumerate()
            .map(|(offset, byte)| ((MEMORY_AS, RHS_PTR + pointer_offset + offset as u32), byte)),
    );
    VmExe::new(program).with_init_memory(memory)
}

fn fixture(equal: bool) -> VmExe<BabyBear> {
    fixture_with_pointer_offset(equal, 0)
}

#[test]
fn checkpoint_execution_preserves_int256_branch_outcomes() {
    let config = Int256Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config).unwrap();
    for (equal, expected_pc, expected_branch_replay_value) in [(false, 8, 0u64), (true, 12, 1u64)] {
        let checkpoint = executor.preflight_instance(&fixture(equal)).unwrap();
        let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let execution = checkpoint
            .execute_from_state(state, PreflightLimits::new(3, 5, 1))
            .unwrap();
        assert_eq!(execution.to_state.byte_pc(), expected_pc);
        assert_eq!(execution.to_state.timestamp, 26);
        let mut expected_replay_values = add_replay_values(equal).to_vec();
        expected_replay_values.push(expected_branch_replay_value);
        assert_eq!(execution.transcript.replay_values, expected_replay_values);
    }
}

#[test]
fn int256_heap_pointers_follow_the_eight_byte_memory_equipartition() {
    let config = Int256Rv64Config {
        system: test_system_config(),
        ..Default::default()
    };
    let executor = VmExecutor::new(config).unwrap();

    for pointer_offset in [0, 8] {
        let exe = fixture_with_pointer_offset(false, pointer_offset);
        let interpreter = executor.interpreter_instance(&exe).unwrap();
        let state = interpreter.create_initial_vm_state(Vec::<Vec<u8>>::new());
        interpreter.execute_from_state(state).unwrap();

        let rvr = executor.instance(&exe).unwrap();
        let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
        rvr.execute_from_state(state).unwrap();
    }

    for pointer_offset in [2, 4, 6] {
        let exe = fixture_with_pointer_offset(false, pointer_offset);
        let interpreter = executor.interpreter_instance(&exe).unwrap();
        let state = interpreter.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let error = match interpreter.execute_from_state(state) {
            Ok(_) => panic!("misaligned Int256 heap pointer unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("eight-byte aligned"), "{error}");

        let rvr = executor.instance(&exe).unwrap();
        let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
        assert!(rvr.execute_from_state(state).is_err());
    }
}
