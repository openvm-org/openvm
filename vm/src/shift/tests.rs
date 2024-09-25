use std::{iter, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::{solve_shift, ShiftChip};
use crate::{
    arch::{
        instructions::Opcode,
        testing::{memory::gen_pointer, MachineChipTestBuilder},
    },
    core::BYTE_XOR_BUS,
    program::Instruction,
};

type F = BabyBear;
const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;

fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    (0..NUM_LIMBS)
        .map(|_| rng.gen_range(0..1 << LIMB_BITS))
        .collect()
}

fn generate_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(rng: &mut StdRng) -> Vec<u32> {
    iter::once(rng.gen_range(0..1 << LIMB_BITS))
        .chain(iter::repeat(0))
        .take(NUM_LIMBS)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_shift_rand_write_execute<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    tester: &mut MachineChipTestBuilder<F>,
    chip: &mut ShiftChip<F, NUM_LIMBS, LIMB_BITS>,
    opcode: Opcode,
    x: Vec<u32>,
    y: Vec<u32>,
    rng: &mut StdRng,
) {
    let address_space_range = || 1usize..=2;

    let d = rng.gen_range(address_space_range());
    let e = rng.gen_range(address_space_range());

    let x_address = gen_pointer(rng, 64);
    let y_address = gen_pointer(rng, 64);
    let res_address = gen_pointer(rng, 64);
    let x_ptr_to_address = gen_pointer(rng, 1);
    let y_ptr_to_address = gen_pointer(rng, 1);
    let res_ptr_to_address = gen_pointer(rng, 1);

    let x_f = x
        .clone()
        .into_iter()
        .map(F::from_canonical_u32)
        .collect::<Vec<_>>();
    let y_f = y
        .clone()
        .into_iter()
        .map(F::from_canonical_u32)
        .collect::<Vec<_>>();

    tester.write_cell(d, x_ptr_to_address, F::from_canonical_usize(x_address));
    tester.write_cell(d, y_ptr_to_address, F::from_canonical_usize(y_address));
    tester.write_cell(d, res_ptr_to_address, F::from_canonical_usize(res_address));
    tester.write::<NUM_LIMBS>(e, x_address, x_f.as_slice().try_into().unwrap());
    tester.write::<NUM_LIMBS>(e, y_address, y_f.as_slice().try_into().unwrap());

    let (z, _, _) = solve_shift::<NUM_LIMBS, LIMB_BITS>(&x, &y, opcode);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode,
            [res_ptr_to_address, x_ptr_to_address, y_ptr_to_address, d, e],
        ),
    );

    assert_eq!(
        z.into_iter().map(F::from_canonical_u32).collect::<Vec<_>>(),
        tester.read::<NUM_LIMBS>(e, res_address)
    )
}

#[test]
fn shift_sll_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_shift::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_shift_rand_write_execute(&mut tester, &mut chip, Opcode::SLL256, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_srl_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_shift::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_shift_rand_write_execute(&mut tester, &mut chip, Opcode::SRL256, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_sra_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_shift::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_shift_rand_write_execute(&mut tester, &mut chip, Opcode::SRA256, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_overflow_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
    );

    let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
    let mut y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
    y[1] = 100;

    run_shift_rand_write_execute(
        &mut tester,
        &mut chip,
        Opcode::SLL256,
        x.clone(),
        y.clone(),
        &mut rng,
    );
    run_shift_rand_write_execute(
        &mut tester,
        &mut chip,
        Opcode::SRL256,
        x.clone(),
        y.clone(),
        &mut rng,
    );
    run_shift_rand_write_execute(
        &mut tester,
        &mut chip,
        Opcode::SRA256,
        x.clone(),
        y.clone(),
        &mut rng,
    );

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn solve_sll_sanity_test() {
    let x: [u32; 32] = [
        45, 7, 61, 186, 49, 53, 119, 68, 145, 55, 102, 126, 9, 195, 23, 26, 197, 216, 251, 31, 74,
        237, 141, 92, 98, 184, 176, 106, 64, 29, 58, 246,
    ];
    let y: [u32; 32] = [
        27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let z: [u32; 32] = [
        0, 0, 0, 104, 57, 232, 209, 141, 169, 185, 35, 138, 188, 49, 243, 75, 24, 190, 208, 40,
        198, 222, 255, 80, 106, 111, 228, 18, 195, 133, 85, 3,
    ];
    let sll_result = solve_shift::<32, 8>(&x, &y, Opcode::SLL256).0;
    for i in 0..32 {
        assert_eq!(z[i], sll_result[i])
    }
}

#[test]
fn solve_srl_sanity_test() {
    let x: [u32; 32] = [
        253, 247, 209, 166, 217, 253, 46, 42, 197, 8, 33, 136, 144, 148, 101, 195, 173, 150, 26,
        215, 233, 90, 213, 185, 119, 255, 238, 174, 31, 190, 221, 72,
    ];
    let y: [u32; 32] = [
        17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let z: [u32; 32] = [
        104, 211, 236, 126, 23, 149, 98, 132, 16, 68, 72, 202, 178, 225, 86, 75, 141, 235, 116,
        173, 234, 220, 187, 127, 119, 215, 15, 223, 110, 36, 0, 0,
    ];
    let srl_result = solve_shift::<32, 8>(&x, &y, Opcode::SRL256).0;
    let sra_result = solve_shift::<32, 8>(&x, &y, Opcode::SRA256).0;
    for i in 0..32 {
        assert_eq!(z[i], srl_result[i]);
        assert_eq!(z[i], sra_result[i]);
    }
}

#[test]
fn solve_sra_sanity_test() {
    let x: [u32; 32] = [
        253, 247, 209, 166, 217, 253, 46, 42, 197, 8, 33, 136, 144, 148, 101, 195, 173, 150, 26,
        215, 233, 90, 213, 185, 119, 255, 238, 174, 31, 190, 221, 200,
    ];
    let y: [u32; 32] = [
        17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let z: [u32; 32] = [
        104, 211, 236, 126, 23, 149, 98, 132, 16, 68, 72, 202, 178, 225, 86, 75, 141, 235, 116,
        173, 234, 220, 187, 127, 119, 215, 15, 223, 110, 228, 255, 255,
    ];
    let sra_result = solve_shift::<32, 8>(&x, &y, Opcode::SRA256).0;
    for i in 0..32 {
        assert_eq!(z[i], sra_result[i])
    }
}
