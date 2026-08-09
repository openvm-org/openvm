use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_poseidon2_air::{Poseidon2Config, POSEIDON2_WIDTH};
use openvm_poseidon2_transpiler::Poseidon2Opcode;
use openvm_stark_backend::{
    interaction::{BusIndex, LookupBus},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, RngCore};

use crate::{
    periphery::{Poseidon2PeripheryAir, Poseidon2PeripheryChip},
    permute::{Poseidon2PermuteAir, Poseidon2PermuteChip, Poseidon2PermuteExecutor},
    POSEIDON2_STATE_BYTES,
};

type F = BabyBear;
/// Harness without Poseidon2Periphery*
type Harness<RA> =
    TestChipHarness<F, Poseidon2PermuteExecutor, Poseidon2PermuteAir, Poseidon2PermuteChip<F>, RA>;
const MAX_TRACE_ROWS: usize = 4096;
/// Dedicated bus index for the adapter-periphery direct lookup. Must not collide with the system
/// bus indices used by `VmChipTestBuilder` (in particular `POSEIDON2_DIRECT_BUS = 6`, which the
/// memory system hasher uses).
const POSEIDON2_PERMUTE_BUS: BusIndex = 13;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Poseidon2PermuteAir,
    Poseidon2PermuteExecutor,
    Poseidon2PermuteChip<F>,
) {
    let executor = Poseidon2PermuteExecutor::new(Poseidon2Opcode::CLASS_OFFSET, address_bits);
    let op_air = Poseidon2PermuteAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        LookupBus::new(POSEIDON2_PERMUTE_BUS),
        address_bits,
        Poseidon2Opcode::CLASS_OFFSET,
    );
    let periphery = Arc::new(Poseidon2PeripheryChip::<F>::new(Poseidon2Config::default()));
    let op_chip = Poseidon2PermuteChip::new(bitwise_chip, address_bits, memory_helper, periphery);
    (op_air, executor, op_chip)
}

struct TestHarness<RA> {
    harness: Harness<RA>,
    bitwise: (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
    perm: (Poseidon2PeripheryAir<F>, Arc<Poseidon2PeripheryChip<F>>),
}

fn create_test_harness<RA: Arena>(tester: &mut VmChipTestBuilder<F>) -> TestHarness<RA> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (op_air, executor, op_chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let periphery_chip = op_chip.periphery.clone();

    let harness = Harness::with_capacity(executor, op_air, op_chip, MAX_TRACE_ROWS);

    let perm_air = Poseidon2PeripheryAir::<F>::new(
        Poseidon2Config::default(),
        LookupBus::new(POSEIDON2_PERMUTE_BUS),
    );

    TestHarness {
        harness,
        bitwise: (bitwise_chip.air, bitwise_chip),
        perm: (perm_air, periphery_chip),
    }
}

fn set_and_execute_single_perm<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    poseidon2_chip: &Poseidon2PeripheryChip<F>,
) {
    let input_words: [F; POSEIDON2_WIDTH] = std::array::from_fn(|_| {
        // Words must be canonical field elements (less than the field characteristic).
        F::from_u32(rng.next_u32() % (1 << 30))
    });
    let expected_out = poseidon2_chip.permute(input_words);

    let rd = gen_pointer(rng, 4);
    let buffer_ptr = gen_pointer(rng, POSEIDON2_STATE_BYTES);
    tester.write(1, rd, (buffer_ptr as u32).to_le_bytes().map(F::from_u8));
    for (i, &word) in input_words.iter().enumerate() {
        let word_bytes = word.as_canonical_u32().to_le_bytes().map(F::from_u8);
        tester.write(2, buffer_ptr + 4 * i, word_bytes);
    }

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(Poseidon2Opcode::PERMUTE.global_opcode(), [rd, 0, 0, 1, 2]),
    );

    let mut output_buffer = [0u8; POSEIDON2_STATE_BYTES];
    for i in 0..POSEIDON2_WIDTH {
        let output_chunk: [F; 4] = tester.read(2, buffer_ptr + 4 * i);
        let output_chunk = output_chunk.map(|x| x.as_canonical_u32() as u8);
        output_buffer[4 * i..4 * i + 4].copy_from_slice(&output_chunk);
    }

    let mut expected_bytes = [0u8; POSEIDON2_STATE_BYTES];
    for (i, word) in expected_out.iter().enumerate() {
        expected_bytes[4 * i..4 * i + 4].copy_from_slice(&word.as_canonical_u32().to_le_bytes());
    }
    assert_eq!(&output_buffer[..], &expected_bytes[..]);
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_poseidon2_positive_tests() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness {
        mut harness,
        bitwise,
        perm,
    } = create_test_harness(&mut tester);

    let num_ops: usize = 100;
    for _ in 0..num_ops {
        set_and_execute_single_perm(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &perm.1,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(perm)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}
