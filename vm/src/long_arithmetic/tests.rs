use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, Rng};

use super::LongAdditionChip;

fn generate_long_number<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    assert!(ARG_SIZE % LIMB_SIZE == 0);
    (0..ARG_SIZE / LIMB_SIZE)
        .map(|_| rng.gen_range(0..1 << LIMB_SIZE))
        .collect()
}
fn generate_long_add_program<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    chip: &mut LongAdditionChip<ARG_SIZE, LIMB_SIZE>,
    len_ops: usize,
) {
    let mut rng = create_seeded_rng();
    let operands = (0..len_ops)
        .map(|_| {
            (
                generate_long_number::<ARG_SIZE, LIMB_SIZE>(&mut rng),
                generate_long_number::<ARG_SIZE, LIMB_SIZE>(&mut rng),
            )
        })
        .collect();
    chip.request(operands);
}

#[test]
fn add_rand_air_test() {
    let len_ops: usize = 15;
    let bus_index: usize = 0;
    let mut chip = LongAdditionChip::<256, 16>::new(bus_index);

    generate_long_add_program(&mut chip, len_ops);

    let trace = chip.generate_trace::<BabyBear>();
    let range_trace = chip.range_checker_gate_chip.generate_trace::<BabyBear>();

    run_simple_test_no_pis(
        vec![&chip.air, &chip.range_checker_gate_chip.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}
