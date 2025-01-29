use std::str::FromStr;

use num_bigint::BigUint;
use num_traits::FromPrimitive;
use openvm_circuit::arch::{testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
};
use openvm_ecc_transpiler::Rv32EdwardsOpcode;
use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_mod_circuit_builder::{test_utils::biguint_to_limbs, ExprBuilderConfig, FieldExpr};
use openvm_rv32_adapters::{rv32_write_heap_default, Rv32VecHeapAdapterChip};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::TeAddChip;

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;
type F = BabyBear;

lazy_static::lazy_static! {
    pub static ref SampleEcPoints: Vec<(BigUint, BigUint)> = {
        // Base point of edwards25519
        let x1 = BigUint::from_str(
            "15112221349535400772501151409588531511454012693041857206046113283949847762202",
        )
        .unwrap();
        let y1 = BigUint::from_str(
            "46316835694926478169428394003475163141307993866256225615783033603165251855960",
        )
        .unwrap();

        // random point on edwards25519
        let x2 = BigUint::from_u32(2).unwrap();
        let y2 = BigUint::from_str(
            "11879831548380997166425477238087913000047176376829905612296558668626594440753",
        )
        .unwrap();

        // This is the sum of (x1, y1) and (x2, y2).
        let x3 = BigUint::from_str(
            "44969869612046584870714054830543834361257841801051546235130567688769346152934",
        )
        .unwrap();
        let y3 = BigUint::from_str(
            "50796027728050908782231253190819121962159170739537197094456293084373503699602",
        )
        .unwrap();

        // This is 2 * (x1, y1)
        let x4 = BigUint::from_str(
            "39226743113244985161159605482495583316761443760287217110659799046557361995496",
        )
        .unwrap();
        let y4 = BigUint::from_str(
            "12570354238812836652656274015246690354874018829607973815551555426027032771563",
        )
        .unwrap();

        vec![(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    };

    pub static ref Edwards25519_Prime: BigUint = BigUint::from_str(
        "57896044618658097711785492504343953926634992332820282019728792003956564819949",
    )
    .unwrap();

    pub static ref Edwards25519_A: BigUint = BigUint::from_str(
        "57896044618658097711785492504343953926634992332820282019728792003956564819948",
    )
    .unwrap();

    pub static ref Edwards25519_D: BigUint = BigUint::from_str(
        "37095705934669439343138083508754565189542113879843219016388785533085940283555",
    )
    .unwrap();

    pub static ref Edwards25519_A_LIMBS: [BabyBear; NUM_LIMBS] =
        big_uint_to_limbs(&Edwards25519_A, LIMB_BITS)
            .into_iter()
            .map(BabyBear::from_canonical_usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    pub static ref Edwards25519_D_LIMBS: [BabyBear; NUM_LIMBS] =
        big_uint_to_limbs(&Edwards25519_D, LIMB_BITS)
            .into_iter()
            .map(BabyBear::from_canonical_usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
}

fn prime_limbs(expr: &FieldExpr) -> Vec<BabyBear> {
    expr.prime_limbs
        .iter()
        .map(|n| BabyBear::from_canonical_usize(*n))
        .collect::<Vec<_>>()
}

#[test]
fn test_add() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: Edwards25519_Prime.clone(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        tester.address_bits(),
        bitwise_chip.clone(),
    );
    let mut chip = TeAddChip::new(
        adapter,
        config,
        Rv32EdwardsOpcode::CLASS_OFFSET,
        Edwards25519_A.clone(),
        Edwards25519_D.clone(),
        tester.range_checker(),
        tester.offline_memory_mutex_arc(),
    );
    //assert_eq!(chip.0.core.expr().builder.num_variables, 12);
    assert_eq!(chip.0.core.air.expr.builder.num_variables, 12);

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();

    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let r = chip
        .0
        .core
        //.expr()
        .air
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![true]);
    assert_eq!(r.len(), 12);

    let outputs = chip
        .0
        .core
        .air
        .output_indices()
        .iter()
        .map(|i| &r[*i])
        .collect::<Vec<_>>();
    assert_eq!(outputs[0], &SampleEcPoints[2].0);
    assert_eq!(outputs[1], &SampleEcPoints[2].1);

    //let prime_limbs: [BabyBear; NUM_LIMBS] = prime_limbs(chip.0.core.expr()).try_into().unwrap();
    let prime_limbs: [BabyBear; NUM_LIMBS] = prime_limbs(&chip.0.core.air.expr).try_into().unwrap();
    let mut one_limbs = [BabyBear::ONE; NUM_LIMBS];
    one_limbs[0] = BabyBear::ONE;
    let setup_instruction = rv32_write_heap_default(
        &mut tester,
        vec![prime_limbs, *Edwards25519_A_LIMBS],
        vec![*Edwards25519_D_LIMBS],
        chip.0.core.air.offset + Rv32EdwardsOpcode::SETUP_TE_ADD as usize,
    );
    tester.execute(&mut chip, &setup_instruction);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![p2_x_limbs, p2_y_limbs],
        chip.0.core.air.offset + Rv32EdwardsOpcode::TE_ADD as usize,
    );

    tester.execute(&mut chip, &instruction);

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}
