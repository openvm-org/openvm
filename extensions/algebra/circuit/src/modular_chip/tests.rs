use std::{array::from_fn, borrow::BorrowMut};

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::arch::{
    instructions::LocalOpcode,
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    AdapterRuntimeContext, Result, VmAdapterInterface, VmChipWrapper, VmCoreChip,
};
use openvm_circuit_primitives::{
    bigint::utils::{big_uint_to_limbs, secp256k1_coord_prime, secp256k1_scalar_prime},
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, VmOpcode};
use openvm_mod_circuit_builder::{
    test_utils::{biguint_to_limbs, generate_field_element},
    ExprBuilderConfig,
};
use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
use openvm_rv32_adapters::{
    rv32_write_heap_default, write_ptr_reg, Rv32IsEqualModAdapterChip, Rv32VecHeapAdapterChip,
};
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use super::{
    ModularAddSubChip, ModularIsEqualChip, ModularIsEqualCoreAir, ModularIsEqualCoreChip,
    ModularIsEqualCoreCols, ModularIsEqualCoreRecord, ModularMulDivChip,
};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;
type F = BabyBear;

const ADD_LOCAL: usize = Rv32ModularArithmeticOpcode::ADD as usize;
const MUL_LOCAL: usize = Rv32ModularArithmeticOpcode::MUL as usize;

#[test]
fn test_coord_addsub() {
    let opcode_offset = 0;
    let modulus = secp256k1_coord_prime();
    test_addsub(opcode_offset, modulus);
}

#[test]
fn test_scalar_addsub() {
    let opcode_offset = 4;
    let modulus = secp256k1_scalar_prime();
    test_addsub(opcode_offset, modulus);
}

fn test_addsub(opcode_offset: usize, modulus: BigUint) {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    // doing 1xNUM_LIMBS reads and writes
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 1, 1, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = ModularAddSubChip::new(
        adapter,
        config,
        Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset,
        tester.range_checker(),
        tester.offline_memory_mutex_arc(),
    );
    let mut rng = create_seeded_rng();
    let num_tests = 50;
    let mut all_ops = vec![ADD_LOCAL + 2]; // setup
    let mut all_a = vec![modulus.clone()];
    let mut all_b = vec![BigUint::zero()];

    // First loop: generate all random test data.
    for _ in 0..num_tests {
        let a_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut a = BigUint::new(a_digits.clone());
        let b_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut b = BigUint::new(b_digits.clone());

        let op = rng.gen_range(0..2) + ADD_LOCAL; // 0 for add, 1 for sub
        a %= &modulus;
        b %= &modulus;

        all_ops.push(op);
        all_a.push(a);
        all_b.push(b);
    }
    // Second loop: actually run the tests.
    for i in 0..=num_tests {
        let op = all_ops[i];
        let a = all_a[i].clone();
        let b = all_b[i].clone();
        if i > 0 {
            // if not setup
            assert!(a < modulus);
            assert!(b < modulus);
        }
        let expected_answer = match op - ADD_LOCAL {
            0 => (&a + &b) % &modulus,
            1 => (&a + &modulus - &b) % &modulus,
            2 => a.clone() % &modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each biguint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 3 * RV32_REGISTER_NUM_LIMBS;
        let addr_ptr3 = 6 * RV32_REGISTER_NUM_LIMBS;

        let data_as = 2;
        let address1 = 0u32;
        let address2 = 128u32;
        let address3 = (1 << 28) + 1234; // a large memory address to test heap adapter

        write_ptr_reg(&mut tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr3, address3);

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(chip.0.core.air.offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(&mut chip, &instruction);

        let expected_limbs = biguint_to_limbs::<NUM_LIMBS>(expected_answer, LIMB_BITS);
        for (i, expected) in expected_limbs.into_iter().enumerate() {
            let address = address3 as usize + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(expected), read_val);
        }
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_coord_muldiv() {
    let opcode_offset = 0;
    let modulus = secp256k1_coord_prime();
    test_muldiv(opcode_offset, modulus);
}

#[test]
fn test_scalar_muldiv() {
    let opcode_offset = 4;
    let modulus = secp256k1_scalar_prime();
    test_muldiv(opcode_offset, modulus);
}

fn test_muldiv(opcode_offset: usize, modulus: BigUint) {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: modulus.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    // doing 1xNUM_LIMBS reads and writes
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 1, 1, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = ModularMulDivChip::new(
        adapter,
        config,
        Rv32ModularArithmeticOpcode::CLASS_OFFSET + opcode_offset,
        tester.range_checker(),
        tester.offline_memory_mutex_arc(),
    );
    let mut rng = create_seeded_rng();
    let num_tests = 50;
    let mut all_ops = vec![MUL_LOCAL + 2];
    let mut all_a = vec![modulus.clone()];
    let mut all_b = vec![BigUint::zero()];

    // First loop: generate all random test data.
    for _ in 0..num_tests {
        let a_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut a = BigUint::new(a_digits.clone());
        let b_digits: Vec<_> = (0..NUM_LIMBS)
            .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
            .collect();
        let mut b = BigUint::new(b_digits.clone());

        // let op = rng.gen_range(2..4); // 2 for mul, 3 for div
        let op = MUL_LOCAL;
        a %= &modulus;
        b %= &modulus;

        all_ops.push(op);
        all_a.push(a);
        all_b.push(b);
    }
    // Second loop: actually run the tests.
    for i in 0..=num_tests {
        let op = all_ops[i];
        let a = all_a[i].clone();
        let b = all_b[i].clone();
        if i > 0 {
            // if not setup
            assert!(a < modulus);
            assert!(b < modulus);
        }
        let expected_answer = match op - MUL_LOCAL {
            0 => (&a * &b) % &modulus,
            1 => (&a * b.modinv(&modulus).unwrap()) % &modulus,
            2 => a.clone() % &modulus,
            _ => panic!(),
        };

        // Write to memories
        // For each biguint (a, b, r), there are 2 writes:
        // 1. address_ptr which stores the actual address
        // 2. actual address which stores the biguint limbs
        // The write of result r is done in the chip.
        let ptr_as = 1;
        let addr_ptr1 = 0;
        let addr_ptr2 = 12;
        let addr_ptr3 = 24;

        let data_as = 2;
        let address1 = 0;
        let address2 = 128;
        let address3 = 256;

        write_ptr_reg(&mut tester, ptr_as, addr_ptr1, address1);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr2, address2);
        write_ptr_reg(&mut tester, ptr_as, addr_ptr3, address3);

        let a_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(a.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address1 as usize, a_limbs);
        let b_limbs: [BabyBear; NUM_LIMBS] =
            biguint_to_limbs(b.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
        tester.write(data_as, address2 as usize, b_limbs);

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(chip.0.core.air.offset + op),
            addr_ptr3 as isize,
            addr_ptr1 as isize,
            addr_ptr2 as isize,
            ptr_as as isize,
            data_as as isize,
        );
        tester.execute(&mut chip, &instruction);

        let expected_limbs = biguint_to_limbs::<NUM_LIMBS>(expected_answer, LIMB_BITS);
        for (i, expected) in expected_limbs.into_iter().enumerate() {
            let address = address3 as usize + i;
            let read_val = tester.read_cell(data_as, address);
            assert_eq!(BabyBear::from_canonical_u32(expected), read_val);
        }
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}

fn test_is_equal<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>(
    opcode_offset: usize,
    modulus: BigUint,
    num_tests: usize,
) {
    let mut rng = create_seeded_rng();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<LIMB_BITS>::new(bitwise_bus);

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let mut chip = ModularIsEqualChip::<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
        Rv32IsEqualModAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        ),
        ModularIsEqualCoreChip::new(modulus.clone(), bitwise_chip.clone(), opcode_offset),
        tester.offline_memory_mutex_arc(),
    );

    {
        let vec = big_uint_to_limbs(&modulus, LIMB_BITS);
        let modulus_limbs: [F; TOTAL_LIMBS] = std::array::from_fn(|i| {
            if i < vec.len() {
                F::from_canonical_usize(vec[i])
            } else {
                F::ZERO
            }
        });

        let setup_instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
            &mut tester,
            vec![modulus_limbs],
            vec![[F::ZERO; TOTAL_LIMBS]],
            opcode_offset + Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize,
        );
        tester.execute(&mut chip, &setup_instruction);
    }
    for _ in 0..num_tests {
        let b = generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(&modulus, &mut rng);
        let c = if rng.gen_bool(0.5) {
            b
        } else {
            generate_field_element::<TOTAL_LIMBS, LIMB_BITS>(&modulus, &mut rng)
        };

        let instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
            &mut tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            opcode_offset + Rv32ModularArithmeticOpcode::IS_EQ as usize,
        );
        tester.execute(&mut chip, &instruction);
    }

    // Special case where b == c are close to the prime
    let b_vec = big_uint_to_limbs(&modulus, LIMB_BITS);
    let mut b = from_fn(|i| if i < b_vec.len() { b_vec[i] as u32 } else { 0 });
    b[0] -= 1;
    let instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
        &mut tester,
        vec![b.map(F::from_canonical_u32)],
        vec![b.map(F::from_canonical_u32)],
        opcode_offset + Rv32ModularArithmeticOpcode::IS_EQ as usize,
    );
    tester.execute(&mut chip, &instruction);

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_modular_is_equal_1x32() {
    test_is_equal::<1, 32, 32>(17, secp256k1_coord_prime(), 100);
}

#[test]
fn test_modular_is_equal_3x16() {
    test_is_equal::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 100);
}

// Wrapper chip for testing a bad setup row
type BadModularIsEqualChip<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmChipWrapper<
    F,
    Rv32IsEqualModAdapterChip<F, 2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    BadModularIsEqualCoreChip<TOTAL_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// Wrapper chip for testing a bad setup row
struct BadModularIsEqualCoreChip<
    const READ_LIMBS: usize,
    const WRITE_LIMBS: usize,
    const LIMB_BITS: usize,
> {
    chip: ModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>,
}

impl<const READ_LIMBS: usize, const WRITE_LIMBS: usize, const LIMB_BITS: usize>
    BadModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
{
    pub fn new(
        modulus: BigUint,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
    ) -> Self {
        Self {
            chip: ModularIsEqualCoreChip::new(modulus, bitwise_lookup_chip, offset),
        }
    }
}

impl<
        F: PrimeField32,
        I: VmAdapterInterface<F>,
        const READ_LIMBS: usize,
        const WRITE_LIMBS: usize,
        const LIMB_BITS: usize,
    > VmCoreChip<F, I> for BadModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>
where
    I::Reads: Into<[[F; READ_LIMBS]; 2]>,
    I::Writes: From<[[F; WRITE_LIMBS]; 1]>,
{
    type Record = ModularIsEqualCoreRecord<F, READ_LIMBS>;
    type Air = ModularIsEqualCoreAir<READ_LIMBS, WRITE_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        // Override the b_diff_idx to be out of bounds.
        // This will cause lt_marker to be all zeros except a 2.
        // There was a bug in this case which allowed b to be less than N.
        self.chip.execute_instruction(instruction, from_pc, reads)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        <ModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS> as VmCoreChip<F, I>>::get_opcode_name(&self.chip, opcode)
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        <ModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS> as VmCoreChip<F, I>>::generate_trace_row(&self.chip, row_slice, record.clone());
        let row_slice: &mut ModularIsEqualCoreCols<_, READ_LIMBS> = row_slice.borrow_mut();
        // decide which bug to test based on b[0]
        if record.b[0] == F::ONE {
            // test the constraint that c_lt_mark = 2 when is_setup = 1
            row_slice.c_lt_mark = F::ONE;
            row_slice.lt_marker = [F::ZERO; READ_LIMBS];
            row_slice.lt_marker[READ_LIMBS - 1] = F::ONE;
            row_slice.c_lt_diff =
                F::from_canonical_u32(self.chip.air.modulus_limbs[READ_LIMBS - 1])
                    - record.c[READ_LIMBS - 1];
            row_slice.b_lt_diff =
                F::from_canonical_u32(self.chip.air.modulus_limbs[READ_LIMBS - 1])
                    - record.b[READ_LIMBS - 1];
        } else if record.b[0] == F::from_canonical_u32(2) {
            // test the constraint that b[i] = N[i] for all i when prefix_sum is not 1 or
            // lt_marker_sum - is_setup
            row_slice.c_lt_mark = F::from_canonical_u8(2);
            row_slice.lt_marker = [F::ZERO; READ_LIMBS];
            row_slice.lt_marker[READ_LIMBS - 1] = F::from_canonical_u8(2);
            row_slice.c_lt_diff =
                F::from_canonical_u32(self.chip.air.modulus_limbs[READ_LIMBS - 1])
                    - record.c[READ_LIMBS - 1];
        } else if record.b[0] == F::from_canonical_u32(3) {
            // test the constraint that sum_i lt_marker[i] = 2 when is_setup = 1
            row_slice.c_lt_mark = F::from_canonical_u8(2);
            row_slice.lt_marker = [F::ZERO; READ_LIMBS];
            row_slice.lt_marker[READ_LIMBS - 1] = F::from_canonical_u8(2);
            row_slice.lt_marker[0] = F::ONE;
            row_slice.b_lt_diff =
                F::from_canonical_u32(self.chip.air.modulus_limbs[0]) - record.b[0];
            row_slice.c_lt_diff =
                F::from_canonical_u32(self.chip.air.modulus_limbs[READ_LIMBS - 1])
                    - record.c[READ_LIMBS - 1];
        }
    }

    fn air(&self) -> &Self::Air {
        <ModularIsEqualCoreChip<READ_LIMBS, WRITE_LIMBS, LIMB_BITS> as VmCoreChip<F, I>>::air(
            &self.chip,
        )
    }
}

// Test that passes the wrong modulus in the setup instruction.
// This proof should fail to verify.
fn test_is_equal_setup_bad<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
>(
    opcode_offset: usize,
    modulus: BigUint,
    b_val: u32, /* used to select which bug to test. currently only 1, 2, and 3 are supported
                 * (because there are three bugs to test) */
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<LIMB_BITS>::new(bitwise_bus);

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let mut chip = BadModularIsEqualChip::<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>::new(
        Rv32IsEqualModAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        ),
        BadModularIsEqualCoreChip::new(modulus.clone(), bitwise_chip.clone(), opcode_offset),
        tester.offline_memory_mutex_arc(),
    );

    let mut b_limbs = [F::ZERO; TOTAL_LIMBS];
    b_limbs[0] = F::from_canonical_u32(b_val);
    let setup_instruction = rv32_write_heap_default::<TOTAL_LIMBS>(
        &mut tester,
        vec![b_limbs],
        vec![[F::ZERO; TOTAL_LIMBS]],
        opcode_offset + Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize,
    );
    tester.execute(&mut chip, &setup_instruction);

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[should_panic]
#[test]
fn test_modular_is_equal_setup_bad_1_1x32() {
    test_is_equal_setup_bad::<1, 32, 32>(17, secp256k1_coord_prime(), 1);
}

#[should_panic]
#[test]
fn test_modular_is_equal_setup_bad_2_1x32_2() {
    test_is_equal_setup_bad::<1, 32, 32>(17, secp256k1_coord_prime(), 2);
}

#[should_panic]
#[test]
fn test_modular_is_equal_setup_bad_3_1x32() {
    test_is_equal_setup_bad::<1, 32, 32>(17, secp256k1_coord_prime(), 3);
}

#[should_panic]
#[test]
fn test_modular_is_equal_setup_bad_1_3x16() {
    test_is_equal_setup_bad::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 1);
}

#[should_panic]
#[test]
fn test_modular_is_equal_setup_bad_2_3x16() {
    test_is_equal_setup_bad::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 2);
}

#[should_panic]
#[test]
fn test_modular_is_equal_setup_bad_3_3x16() {
    test_is_equal_setup_bad::<3, 16, 48>(17, BLS12_381_MODULUS.clone(), 3);
}
