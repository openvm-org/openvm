use num_bigint::BigUint;
#[cfg(feature = "cuda")]
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
#[cfg(feature = "cuda")]
use openvm_circuit::utils::test_gpu_engine;
use openvm_circuit::{
    arch::{rvr::RvrCheckpointPreflightLimits, VirtualMachine, VmExecutor},
    utils::{test_cpu_engine, test_system_config},
};
use openvm_circuit_primitives::bigint::utils::secp256k1_coord_prime;
#[cfg(feature = "cuda")]
use openvm_instructions::VmOpcode;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    LocalOpcode, SystemOpcode,
};
#[cfg(feature = "cuda")]
use openvm_stark_backend::StarkEngine;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::{modular_is_eq_x0_destination, Rv64ModularConfig, Rv64ModularCpuBuilder};
#[cfg(feature = "cuda")]
use super::{
    AlgebraRvrGpuTracegen, Rv64ModularHybridBuilder, Rv64ModularWithFp2Config,
    Rv64ModularWithFp2HybridBuilder,
};

const SETUP_DST_PTR: u32 = 0x100;
const SUM_PTR: u32 = 0x200;
const MODULUS_PTR: u32 = 0x300;
const LHS_PTR: u32 = 0x400;
const RHS_PTR: u32 = 0x500;

#[cfg(feature = "cuda")]
const MOD_MULDIV_SETUP_DST_PTR: u32 = 0x100;
#[cfg(feature = "cuda")]
const MOD_MUL_DST_PTR: u32 = 0x140;
#[cfg(feature = "cuda")]
const MOD_DIV_DST_PTR: u32 = 0x180;
#[cfg(feature = "cuda")]
const FIELD_EXPR_MODULUS_PTR: u32 = 0x200;
#[cfg(feature = "cuda")]
const MOD_LHS_PTR: u32 = 0x240;
#[cfg(feature = "cuda")]
const MOD_RHS_PTR: u32 = 0x280;
#[cfg(feature = "cuda")]
const FP2_ADDSUB_SETUP_DST_PTR: u32 = 0x300;
#[cfg(feature = "cuda")]
const FP2_ADD_DST_PTR: u32 = 0x380;
#[cfg(feature = "cuda")]
const FP2_SUB_DST_PTR: u32 = 0x400;
#[cfg(feature = "cuda")]
const FP2_MULDIV_SETUP_DST_PTR: u32 = 0x480;
#[cfg(feature = "cuda")]
const FP2_MUL_DST_PTR: u32 = 0x500;
#[cfg(feature = "cuda")]
const FP2_DIV_DST_PTR: u32 = 0x580;
#[cfg(feature = "cuda")]
const FP2_LHS_PTR: u32 = 0x600;
#[cfg(feature = "cuda")]
const FP2_RHS_PTR: u32 = 0x680;

fn reg(index: usize) -> usize {
    index * RV64_REGISTER_BYTES as usize
}

fn padded_bytes(value: &BigUint) -> [u8; 32] {
    let bytes = value.to_bytes_le();
    assert!(bytes.len() <= 32);
    std::array::from_fn(|index| bytes.get(index).copied().unwrap_or_default())
}

fn fixture_with_pointer_offset(pointer_offset: u32) -> (Program<BabyBear>, VmExe<BabyBear>) {
    let instructions = [
        Instruction::from_usize(
            Rv64ModularArithmeticOpcode::SETUP_ADDSUB.global_opcode(),
            [
                reg(1),
                reg(2),
                reg(0),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(
            Rv64ModularArithmeticOpcode::ADD.global_opcode(),
            [
                reg(3),
                reg(4),
                reg(5),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(
            Rv64ModularArithmeticOpcode::SETUP_ISEQ.global_opcode(),
            [
                reg(6),
                reg(2),
                reg(0),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(
            Rv64ModularArithmeticOpcode::IS_EQ.global_opcode(),
            [
                reg(7),
                reg(4),
                reg(4),
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);

    let mut memory = SparseMemoryImage::default();
    for (register, pointer) in [
        (1, SETUP_DST_PTR),
        (2, MODULUS_PTR),
        (3, SUM_PTR),
        (4, LHS_PTR),
        (5, RHS_PTR),
    ] {
        let pointer = pointer + pointer_offset;
        memory.extend(
            u64::from(pointer)
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)),
        );
    }
    for (pointer, value) in [
        (MODULUS_PTR, padded_bytes(&secp256k1_coord_prime())),
        (LHS_PTR, padded_bytes(&BigUint::from(5u32))),
        (RHS_PTR, padded_bytes(&BigUint::from(7u32))),
    ] {
        let pointer = pointer + pointer_offset;
        memory.extend(
            value
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((RV64_MEMORY_AS, pointer + offset as u32), byte)),
        );
    }

    (
        program.clone(),
        VmExe::new(program).with_init_memory(memory),
    )
}

fn fixture() -> (Program<BabyBear>, VmExe<BabyBear>) {
    fixture_with_pointer_offset(0)
}

fn config() -> Rv64ModularConfig {
    let mut config = Rv64ModularConfig::new(vec![secp256k1_coord_prime()]);
    config.system = test_system_config();
    config
}

#[cfg(feature = "cuda")]
fn write_fixture_bytes(
    memory: &mut SparseMemoryImage,
    address_space: u32,
    address: u32,
    bytes: impl IntoIterator<Item = u8>,
) {
    memory.extend(
        bytes
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((address_space, address + offset as u32), byte)),
    );
}

#[cfg(feature = "cuda")]
fn write_fixture_pointer(memory: &mut SparseMemoryImage, register: usize, pointer: u32) {
    write_fixture_bytes(
        memory,
        RV64_REGISTER_AS,
        reg(register) as u32,
        u64::from(pointer).to_le_bytes(),
    );
}

#[cfg(feature = "cuda")]
fn field_expr_instruction(
    opcode: VmOpcode,
    destination: usize,
    lhs: usize,
    rhs: usize,
) -> Instruction<BabyBear> {
    Instruction::from_usize(
        opcode,
        [
            reg(destination),
            reg(lhs),
            reg(rhs),
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

#[cfg(feature = "cuda")]
fn field_expr_fixture(modulus: &BigUint) -> (Program<BabyBear>, VmExe<BabyBear>) {
    let instructions = [
        field_expr_instruction(
            Rv64ModularArithmeticOpcode::SETUP_MULDIV.global_opcode(),
            1,
            2,
            0,
        ),
        field_expr_instruction(Rv64ModularArithmeticOpcode::MUL.global_opcode(), 3, 4, 5),
        field_expr_instruction(Rv64ModularArithmeticOpcode::DIV.global_opcode(), 6, 4, 5),
        field_expr_instruction(Fp2Opcode::SETUP_ADDSUB.global_opcode(), 7, 2, 0),
        field_expr_instruction(Fp2Opcode::ADD.global_opcode(), 8, 9, 10),
        field_expr_instruction(Fp2Opcode::SUB.global_opcode(), 11, 9, 10),
        field_expr_instruction(Fp2Opcode::SETUP_MULDIV.global_opcode(), 12, 2, 0),
        field_expr_instruction(Fp2Opcode::MUL.global_opcode(), 13, 9, 10),
        field_expr_instruction(Fp2Opcode::DIV.global_opcode(), 14, 9, 10),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
    ];
    let program = Program::from_instructions(&instructions);
    let mut memory = SparseMemoryImage::default();

    for (register, pointer) in [
        (1, MOD_MULDIV_SETUP_DST_PTR),
        (2, FIELD_EXPR_MODULUS_PTR),
        (3, MOD_MUL_DST_PTR),
        (4, MOD_LHS_PTR),
        (5, MOD_RHS_PTR),
        (6, MOD_DIV_DST_PTR),
        (7, FP2_ADDSUB_SETUP_DST_PTR),
        (8, FP2_ADD_DST_PTR),
        (9, FP2_LHS_PTR),
        (10, FP2_RHS_PTR),
        (11, FP2_SUB_DST_PTR),
        (12, FP2_MULDIV_SETUP_DST_PTR),
        (13, FP2_MUL_DST_PTR),
        (14, FP2_DIV_DST_PTR),
    ] {
        write_fixture_pointer(&mut memory, register, pointer);
    }

    write_fixture_bytes(
        &mut memory,
        RV64_MEMORY_AS,
        FIELD_EXPR_MODULUS_PTR,
        padded_bytes(modulus),
    );
    write_fixture_bytes(
        &mut memory,
        RV64_MEMORY_AS,
        MOD_LHS_PTR,
        padded_bytes(&BigUint::from(4u32)),
    );
    write_fixture_bytes(
        &mut memory,
        RV64_MEMORY_AS,
        MOD_RHS_PTR,
        padded_bytes(&BigUint::from(2u32)),
    );
    write_fixture_bytes(
        &mut memory,
        RV64_MEMORY_AS,
        FP2_LHS_PTR,
        padded_bytes(&(BigUint::from(1u32) << 32))
            .into_iter()
            .chain([0; 32]),
    );
    write_fixture_bytes(
        &mut memory,
        RV64_MEMORY_AS,
        FP2_RHS_PTR,
        padded_bytes(&BigUint::from(1u32))
            .into_iter()
            .chain(padded_bytes(&BigUint::from(1u32))),
    );

    (
        program.clone(),
        VmExe::new(program).with_init_memory(memory),
    )
}

#[cfg(feature = "cuda")]
fn field_expr_config(modulus: BigUint) -> Rv64ModularWithFp2Config {
    let mut config = Rv64ModularWithFp2Config::new(vec![("FieldExprTest".to_string(), modulus)]);
    config.modular.system = test_system_config();
    config
}

#[cfg(feature = "cuda")]
fn prove_field_expr_checkpoint_replay(modulus: BigUint) {
    let (program, exe) = field_expr_fixture(&modulus);
    let config = field_expr_config(modulus);
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) = VirtualMachine::new_with_keygen(
        test_gpu_engine(),
        Rv64ModularWithFp2HybridBuilder,
        config.clone(),
    )
    .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let execution = checkpoint
        .execute_from_state(state, RvrCheckpointPreflightLimits::new(10, 256, 1))
        .unwrap();
    assert_eq!(execution.retired, 10);

    let gpu_program = AlgebraRvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.modular.system.memory_config,
        &config.modular.modular,
        Some(&config.fp2),
        &vm.engine.device().device_ctx,
    )
    .unwrap();
    let (transcript, replay_plan) = AlgebraRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    assert_eq!(transcript.error_code().unwrap(), 0);
    for opcode in [
        Rv64ModularArithmeticOpcode::SETUP_MULDIV.global_opcode(),
        Rv64ModularArithmeticOpcode::MUL.global_opcode(),
        Rv64ModularArithmeticOpcode::DIV.global_opcode(),
        Fp2Opcode::SETUP_ADDSUB.global_opcode(),
        Fp2Opcode::ADD.global_opcode(),
        Fp2Opcode::SUB.global_opcode(),
        Fp2Opcode::SETUP_MULDIV.global_opcode(),
        Fp2Opcode::MUL.global_opcode(),
        Fp2Opcode::DIV.global_opcode(),
    ] {
        assert_eq!(replay_plan.opcode_range(opcode).len(), 1);
    }

    let tracegen = AlgebraRvrGpuTracegen::new(
        &gpu_program,
        &transcript,
        &replay_plan,
        &config.modular.modular,
        Some(&config.fp2),
    )
    .unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[test]
fn modular_checkpoint_executor_records_only_irreducible_results() {
    let (_, exe) = fixture();
    let executor = VmExecutor::new(config()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let execution = checkpoint
        .execute_from_state(state, RvrCheckpointPreflightLimits::new(5, 5, 1))
        .unwrap();

    // SETUP_ADDSUB and SETUP_ISEQ are derivable without residuals. ADD needs
    // four output words and IS_EQ needs one result bit.
    assert_eq!(execution.retired, 5);
    assert_eq!(execution.to_state.pc, 16);
    assert_eq!(execution.to_state.timestamp, 53);
    assert_eq!(execution.transcript.residuals, [12, 0, 0, 0, 1]);
}

#[test]
fn modular_metering_counts_only_irreducible_results() {
    let (_, exe) = fixture();
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ModularCpuBuilder, config())
            .unwrap();
    let metered_ctx = vm.build_metered_ctx(&exe);
    let (segments, _) = vm
        .metered_instance(&exe)
        .unwrap()
        .execute_metered(Vec::<Vec<u8>>::new(), metered_ctx)
        .unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].num_insns, 5);
    assert_eq!(segments[0].num_checkpoint_residuals, 5);
}

#[test]
fn modular_is_equal_rejects_x0_destination_before_execution() {
    for opcode in [
        Rv64ModularArithmeticOpcode::IS_EQ,
        Rv64ModularArithmeticOpcode::SETUP_ISEQ,
    ] {
        let program = Program::from_instructions(&[
            Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    reg(0),
                    reg(1),
                    reg(2),
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                ],
            ),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
        ]);
        assert_eq!(modular_is_eq_x0_destination(&program, 1), Some(0));
        let exe = VmExe::new(program);
        let executor = VmExecutor::new(config()).unwrap();
        assert!(executor.interpreter_instance(&exe).is_err());
        assert!(executor
            .rvr_experimental_checkpoint_preflight_instance(&exe, None)
            .is_err());
    }
}

#[test]
fn modular_heap_pointers_follow_the_eight_byte_memory_equipartition() {
    let executor = VmExecutor::new(config()).unwrap();

    for pointer_offset in [0, 8] {
        let (_, exe) = fixture_with_pointer_offset(pointer_offset);
        let interpreter = executor.interpreter_instance(&exe).unwrap();
        let state = interpreter.create_initial_vm_state(Vec::<Vec<u8>>::new());
        interpreter.execute_from_state(state).unwrap();

        let rvr = executor.rvr_instance(&exe, None).unwrap();
        let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
        rvr.execute_from_state(state).unwrap();
    }

    for pointer_offset in [2, 4, 6] {
        let (_, exe) = fixture_with_pointer_offset(pointer_offset);
        let interpreter = executor.interpreter_instance(&exe).unwrap();
        let state = interpreter.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let error = match interpreter.execute_from_state(state) {
            Ok(_) => panic!("misaligned modular heap pointer unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("eight-byte aligned"), "{error}");

        let rvr = executor.rvr_instance(&exe, None).unwrap();
        let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
        assert!(rvr.execute_from_state(state).is_err());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn modular_checkpoint_expansion_proves_without_records() {
    let (program, exe) = fixture();
    let config = config();
    let executor = VmExecutor::new(config.clone()).unwrap();
    let checkpoint = executor
        .rvr_experimental_checkpoint_preflight_instance(&exe, None)
        .unwrap();
    let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
    let (mut vm, pk) = VirtualMachine::new_with_keygen(
        test_gpu_engine(),
        Rv64ModularHybridBuilder,
        config.clone(),
    )
    .unwrap();
    let cached_program = vm.commit_program_on_device(&program);
    vm.load_program(cached_program);
    vm.transport_init_memory_to_device(&state.memory);
    let mut execution = checkpoint
        .execute_from_state(state, RvrCheckpointPreflightLimits::new(5, 5, 1))
        .unwrap();
    assert_eq!(execution.retired, 5);
    assert_eq!(execution.to_state.timestamp, 53);
    assert_eq!(execution.transcript.residuals, [12, 0, 0, 0, 1]);

    let gpu_program = AlgebraRvrGpuTracegen::upload_checkpoint_program(
        &program,
        &config.system.memory_config,
        &config.modular,
        None,
        &vm.engine.device().device_ctx,
    )
    .unwrap();

    let missing = execution.transcript.residuals.pop().unwrap();
    let error = AlgebraRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .err()
    .expect("missing Algebra residual must fail checkpoint replay");
    assert!(error.to_string().contains("code 306"), "{error}");
    execution.transcript.residuals.push(missing);

    let (transcript, replay_plan) = AlgebraRvrGpuTracegen::expand_checkpoint_replay(
        &vm,
        &gpu_program,
        &execution,
        execution.retired,
    )
    .unwrap();
    assert_eq!(transcript.error_code().unwrap(), 0);
    assert_eq!(transcript.memory_log_host().unwrap().len(), 52);
    for opcode in [
        Rv64ModularArithmeticOpcode::SETUP_ADDSUB,
        Rv64ModularArithmeticOpcode::ADD,
        Rv64ModularArithmeticOpcode::SETUP_ISEQ,
        Rv64ModularArithmeticOpcode::IS_EQ,
    ] {
        assert_eq!(replay_plan.opcode_range(opcode.global_opcode()).len(), 1);
    }

    let tracegen = AlgebraRvrGpuTracegen::new(
        &gpu_program,
        &transcript,
        &replay_plan,
        &config.modular,
        None,
    )
    .unwrap();
    let proving_ctx = tracegen.generate_proving_ctx(&mut vm).unwrap();
    drop(replay_plan);
    drop(transcript);
    let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
    vm.engine.verify(&pk.get_vk(), &proof).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn field_expr_checkpoint_replay_proves_standard_prime_on_gpu() {
    prove_field_expr_checkpoint_replay(secp256k1_coord_prime());
}

#[cfg(feature = "cuda")]
#[test]
fn field_expr_checkpoint_replay_proves_composite_modulus_with_cpu_projection() {
    prove_field_expr_checkpoint_replay(BigUint::from(15u32));
}
