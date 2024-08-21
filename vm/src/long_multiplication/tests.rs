use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, Rng};

use super::LongMultiplicationChip;
use crate::cpu::OpCode;

type F = BabyBear;

fn generate_long_number(rng: &mut StdRng, arg_size: usize, limb_size: usize) -> Vec<u32> {
    assert!(arg_size % limb_size == 0);
    (0..arg_size / limb_size)
        .map(|_| rng.gen_range(0..1 << limb_size))
        .collect()
}

fn generate_mul_program(
    chip: &mut LongMultiplicationChip,
    len_ops: usize,
    arg_size: usize,
    limb_size: usize,
) {
    let mut rng = create_seeded_rng();
    let opcodes = vec![OpCode::MUL256; len_ops];
    let operands = (0..len_ops)
        .map(|_| {
            (
                generate_long_number(&mut rng, arg_size, limb_size),
                generate_long_number(&mut rng, arg_size, limb_size),
            )
        })
        .collect();
    chip.request(opcodes, operands);
}

#[test]
fn long_mul_rand_air_test() {
    let len_ops: usize = 15;
    let arg_size: usize = 256;
    let limb_size: usize = 8;
    let bus_index: usize = 0;
    let mut chip = LongMultiplicationChip::new(bus_index, arg_size, limb_size, OpCode::MUL256);

    generate_mul_program(&mut chip, len_ops, arg_size, limb_size);

    let trace = chip.generate_trace::<F>();
    let range_trace = chip.range_checker_chip.generate_trace::<F>();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker_chip.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}
