use std::{array, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::{core::solve_shift, Rv32ShiftChip, ShiftCoreChip};
use crate::{
    arch::{
        instructions::ShiftOpcode,
        testing::{memory::gen_pointer, VmChipTestBuilder},
        InstructionExecutor,
    },
    kernels::core::BYTE_XOR_BUS,
    rv32im::adapters::{
        Rv32BaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LANES, RV_IS_TYPE_IMM_BITS,
    },
    system::program::Instruction,
};

type F = BabyBear;

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|_| rng.gen_range(0..(1 << LIMB_BITS)))
}

fn generate_rv32_immediate(rng: &mut StdRng) -> (Option<usize>, [u32; RV32_REGISTER_NUM_LANES]) {
    let mut imm: u32 = rng.gen_range(0..(1 << RV_IS_TYPE_IMM_BITS));
    if (imm & 0x800) != 0 {
        imm |= !0xFFF
    }
    (
        Some((imm & 0xFFFFFF) as usize),
        [
            imm as u8,
            (imm >> 8) as u8,
            (imm >> 16) as u8,
            (imm >> 16) as u8,
        ]
        .map(|x| x as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_rv32_shift_rand_write_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: ShiftOpcode,
    b: [u32; RV32_REGISTER_NUM_LANES],
    c: [u32; RV32_REGISTER_NUM_LANES],
    c_imm: Option<usize>,
    rng: &mut StdRng,
) {
    let is_imm = c_imm.is_some();

    let rs1 = gen_pointer(rng, 32);
    let rs2 = c_imm.unwrap_or_else(|| gen_pointer(rng, 32));
    let rd = gen_pointer(rng, 32);

    tester.write::<RV32_REGISTER_NUM_LANES>(1, rs1, b.map(F::from_canonical_u32));
    if !is_imm {
        tester.write::<RV32_REGISTER_NUM_LANES>(1, rs2, c.map(F::from_canonical_u32));
    }

    let (a, _, _) = solve_shift::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(opcode, &b, &c);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [rd, rs1, rs2, 1, if is_imm { 0 } else { 1 }],
        ),
    );

    assert_eq!(
        a.map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LANES>(1, rd)
    );
}

fn run_rv32_shift_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32ShiftChip::<F>::new(
        Rv32BaseAluAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        ShiftCoreChip::new(
            xor_lookup_chip.clone(),
            tester.memory_controller().borrow().range_checker.clone(),
            0,
        ),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let b = generate_long_number::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(&mut rng);
        let (c_imm, c) = if rng.gen_bool(0.5) {
            (
                None,
                generate_long_number::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(&mut rng),
            )
        } else {
            generate_rv32_immediate(&mut rng)
        };
        run_rv32_shift_rand_write_execute(&mut tester, &mut chip, opcode, b, c, c_imm, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_shift_sll_rand_test() {
    run_rv32_shift_rand_test(ShiftOpcode::SLL, 12);
}

#[test]
fn rv32_shift_srl_rand_test() {
    run_rv32_shift_rand_test(ShiftOpcode::SRL, 12);
}

#[test]
fn rv32_shift_sra_rand_test() {
    run_rv32_shift_rand_test(ShiftOpcode::SRA, 12);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adapter is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

// TODO: negative tests for shift

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
///
#[test]
fn solve_sll_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LANES] = [45, 7, 61, 186];
    let y: [u32; RV32_REGISTER_NUM_LANES] = [27, 0, 0, 0];
    let z: [u32; RV32_REGISTER_NUM_LANES] = [0, 0, 0, 104];
    let (result, limb_shift, bit_shift) =
        solve_shift::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(ShiftOpcode::SLL, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LANES {
        assert_eq!(z[i], result[i])
    }
    assert_eq!((y[0] as usize) / RV32_CELL_BITS, limb_shift);
    assert_eq!((y[0] as usize) % RV32_CELL_BITS, bit_shift);
}

#[test]
fn solve_srl_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LANES] = [31, 190, 221, 200];
    let y: [u32; RV32_REGISTER_NUM_LANES] = [17, 0, 0, 0];
    let z: [u32; RV32_REGISTER_NUM_LANES] = [110, 100, 0, 0];
    let (result, limb_shift, bit_shift) =
        solve_shift::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(ShiftOpcode::SRL, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LANES {
        assert_eq!(z[i], result[i])
    }
    assert_eq!((y[0] as usize) / RV32_CELL_BITS, limb_shift);
    assert_eq!((y[0] as usize) % RV32_CELL_BITS, bit_shift);
}

#[test]
fn solve_sra_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LANES] = [31, 190, 221, 200];
    let y: [u32; RV32_REGISTER_NUM_LANES] = [17, 0, 0, 0];
    let z: [u32; RV32_REGISTER_NUM_LANES] = [110, 228, 255, 255];
    let (result, limb_shift, bit_shift) =
        solve_shift::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(ShiftOpcode::SRA, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LANES {
        assert_eq!(z[i], result[i])
    }
    assert_eq!((y[0] as usize) / RV32_CELL_BITS, limb_shift);
    assert_eq!((y[0] as usize) % RV32_CELL_BITS, bit_shift);
}
