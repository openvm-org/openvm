use num_bigint::BigUint;
use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
use openvm_circuit::{
    arch::{rvr::RvrCheckpointPreflightLimits, VmExecutor},
    utils::test_system_config,
};
use openvm_circuit_primitives::bigint::utils::secp256k1_coord_prime;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    LocalOpcode, SystemOpcode,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::Rv64ModularConfig;

const SETUP_DST_PTR: u32 = 0x100;
const SUM_PTR: u32 = 0x200;
const MODULUS_PTR: u32 = 0x300;
const LHS_PTR: u32 = 0x400;
const RHS_PTR: u32 = 0x500;

fn reg(index: usize) -> usize {
    index * RV64_REGISTER_BYTES as usize
}

fn padded_bytes(value: &BigUint) -> [u8; 32] {
    let bytes = value.to_bytes_le();
    assert!(bytes.len() <= 32);
    std::array::from_fn(|index| bytes.get(index).copied().unwrap_or_default())
}

fn fixture() -> (Program<BabyBear>, VmExe<BabyBear>) {
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

fn config() -> Rv64ModularConfig {
    let mut config = Rv64ModularConfig::new(vec![secp256k1_coord_prime()]);
    config.system = test_system_config();
    config
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
        let exe = VmExe::new(program);
        let executor = VmExecutor::new(config()).unwrap();
        assert!(executor.interpreter_instance(&exe).is_err());
        assert!(executor
            .rvr_experimental_checkpoint_preflight_instance(&exe, None)
            .is_err());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn modular_checkpoint_expansion_proves_without_records() {
    use openvm_circuit::{arch::VirtualMachine, utils::test_gpu_engine};
    use openvm_stark_backend::StarkEngine;

    use super::{AlgebraRvrGpuTracegen, Rv64ModularHybridBuilder};

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
