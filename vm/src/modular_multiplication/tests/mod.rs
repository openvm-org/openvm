use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;
use afs_test_utils::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::RngCore;

use crate::{
    program::Program,
    vm::{
        config::{DEFAULT_MAX_SEGMENT_LEN, MemoryConfig, VmConfig},
        VirtualMachine,
    },
};
use crate::arch::instructions::Opcode::SECP256K1_COORD_MUL;
use crate::cpu::trace::Instruction;
use crate::modular_multiplication::{bigint_to_elems, ModularArithmeticChip};

fn make_vm(
    program: Program<BabyBear>,
    field_arithmetic_enabled: bool,
    field_extension_enabled: bool,
) -> VirtualMachine<BabyBear> {
    VirtualMachine::<BabyBear>::new(
        VmConfig {
            field_arithmetic_enabled,
            field_extension_enabled,
            compress_poseidon2_enabled: false,
            perm_poseidon2_enabled: false,
            modular_multiplication_enabled: true,
            is_less_than_enabled: false,
            memory_config: MemoryConfig {
                addr_space_max_bits: 16,
                pointer_max_bits: 16,
                clk_max_bits: 16,
                decomp: 16,
            },
            num_public_values: 4,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            collect_metrics: false,
        },
        program,
        vec![],
    )
}

#[test]
fn test_modular_multiplication_runtime() {
    let mut vm = make_vm(
        Program {
            instructions: vec![],
            debug_infos: vec![],
        },
        true,
        true,
    );
    assert_eq!(vm.segments.len(), 1);
    let segment = &mut vm.segments[0];

    let num_digits = 8;

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);
    // if these are not true then trace is not guaranteed to be verifiable
    let modulus = ModularArithmeticBigIntAir::secp256k1_coord_prime();
    assert!(a < modulus);
    assert!(b < modulus);

    let r = (a.clone() * b.clone()) % modulus.clone();

    let address1 = 0;
    let address2 = 100;
    let address3 = 4000;
    let air = &segment.modular_arithmetic_chips.get(&modulus).unwrap().air;
    let repr_bits = air.air.repr_bits;
    let num_elems = air.air.limb_dimensions.io_limb_sizes.len();

    for (i, &elem) in bigint_to_elems(a, repr_bits, num_elems).iter().enumerate() {
        let address = address1 + i;
        segment.memory_chip.borrow_mut().write_word(
            BabyBear::one(),
            BabyBear::from_canonical_usize(address),
            [elem],
        );
    }
    for (i, &elem) in bigint_to_elems(b, repr_bits, num_elems).iter().enumerate() {
        let address = address2 + i;
        segment.memory_chip.borrow_mut().write_word(
            BabyBear::one(),
            BabyBear::from_canonical_usize(address),
            [elem],
        );
    }
    ModularArithmeticChip::calculate(
        segment,
        Instruction::from_isize(
            SECP256K1_COORD_MUL,
            address1 as isize,
            address2 as isize,
            address3 as isize,
            0,
            1,
        ),
    );
    for (i, &elem) in bigint_to_elems(r, repr_bits, num_elems).iter().enumerate() {
        let address = address3 + i;
        segment.memory_chip.borrow_mut().write_word(
            BabyBear::one(),
            BabyBear::from_canonical_usize(address),
            [elem],
        );
    }
}
