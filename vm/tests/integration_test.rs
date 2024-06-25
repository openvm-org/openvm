use std::sync::Arc;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;

use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::CpuChip;
use stark_vm::cpu::OpCode::*;
use stark_vm::cpu::RANGE_CHECKER_BUS;
use stark_vm::field_arithmetic::FieldArithmeticAir;
use stark_vm::memory::offline_checker::OfflineChecker;
use stark_vm::memory::MemoryAccess;
use stark_vm::program::ProgramAir;

const DATA_LEN: usize = 1;
const ADDR_SPACE_LIMB_BITS: usize = 8;
const POINTER_LIMB_BITS: usize = 8;
const CLK_LIMB_BITS: usize = 8;
const DECOMP: usize = 4;
const RANGE_MAX: u32 = 1 << DECOMP;

const MEMORY_TRACE_DEGREE: usize = 32;

fn air_test(is_field_arithmetic_enabled: bool, program: Vec<Instruction<BabyBear>>) {
    let cpu_chip = CpuChip::new(is_field_arithmetic_enabled);
    let execution = cpu_chip.generate_program_execution(program);

    let program_air = ProgramAir::new(execution.program.clone());
    let program_trace = program_air.generate_trace(&execution);

    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, RANGE_MAX));
    let offline_checker = OfflineChecker::new(
        DATA_LEN,
        ADDR_SPACE_LIMB_BITS,
        POINTER_LIMB_BITS,
        CLK_LIMB_BITS,
        DECOMP,
    );

    let ops = execution
        .memory_accesses
        .iter()
        .map(|access| MemoryAccess {
            address: access.address,
            op_type: access.op_type,
            address_space: access.address_space,
            timestamp: access.timestamp,
            data: vec![access.data],
        })
        .collect::<Vec<MemoryAccess<BabyBear>>>();
    let offline_checker_trace =
        offline_checker.generate_trace(ops, range_checker.clone(), MEMORY_TRACE_DEGREE);

    let range_trace = range_checker.generate_trace();
    println!("range_trace: {:?}", range_trace);

    let field_arithmetic_air = FieldArithmeticAir::new();
    let field_arithmetic_trace = field_arithmetic_air.generate_trace(&execution);

    let test_result = if is_field_arithmetic_enabled {
        run_simple_test_no_pis(
            vec![
                &cpu_chip.air,
                &program_air,
                &offline_checker,
                &field_arithmetic_air,
                &range_checker.air,
            ],
            vec![
                execution.trace(),
                program_trace,
                offline_checker_trace,
                field_arithmetic_trace,
                range_trace,
            ],
        )
    } else {
        run_simple_test_no_pis(
            vec![
                &cpu_chip.air,
                &program_air,
                &offline_checker,
                &range_checker.air,
            ],
            vec![
                execution.trace(),
                program_trace,
                offline_checker_trace,
                range_trace,
            ],
        )
    };

    test_result.expect("Verification failed");
}

#[test]
fn test_vm_1() {
    let n = 2;
    /*
    Instruction 0 assigns word[0]_1 to n.
    Instruction 4 terminates
    The remainder is a loop that decrements word[0]_1 until it reaches 0, then terminates.
    Instruction 1 checks if word[0]_1 is 0 yet, and if so sets pc to 5 in order to terminate
    Instruction 2 decrements word[0]_1 (using word[1]_1)
    Instruction 3 uses JAL as a simple jump to go back to instruction 1 (repeating the loop).
     */
    let program = vec![
        // word[0]_1 <- word[n]_0
        Instruction::from_isize(STOREW, n, 0, 0, 0, 1),
        // if word[0]_1 == 0 then pc += 3
        Instruction::from_isize(BEQ, 0, 0, 3, 1, 0),
        // word[0]_1 <- word[0]_1 - word[1]_0
        Instruction::from_isize(FSUB, 0, 0, 1, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
    ];

    air_test(true, program);
}

#[test]
fn test_vm_without_field_arithmetic() {
    let field_arithmetic_enabled = false;

    /*
    Instruction 0 assigns word[0]_1 to 5.
    Instruction 1 checks if word[0]_1 is *not* 4, and if so jumps to instruction 4.
    Instruction 2 is never run.
    Instruction 3 terminates.
    Instruction 4 checks if word[0]_1 is 5, and if so jumps to instruction 3 to terminate.
     */
    let program = vec![
        // word[0]_1 <- word[5]_0
        Instruction::from_isize(STOREW, 5, 0, 0, 0, 1),
        // if word[0]_1 != 4 then pc += 2
        Instruction::from_isize(BNE, 0, 4, 3, 1, 0),
        // word[2]_1 <- pc + 1, pc -= 2
        Instruction::from_isize(JAL, 2, -2, 0, 1, 0),
        // terminate
        Instruction::from_isize(TERMINATE, 0, 0, 0, 0, 0),
        // if word[0]_1 == 5 then pc -= 1
        Instruction::from_isize(BEQ, 0, 5, -1, 1, 0),
    ];

    air_test(field_arithmetic_enabled, program);
}
