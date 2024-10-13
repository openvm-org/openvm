use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::{rngs::StdRng, Rng};

use super::{Rv32AuipcChip, Rv32AuipcCoreChip};
use crate::{
    arch::{
        instructions::{
            Rv32AuipcOpcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder},
        Rv32RdWriteAdapter,
    },
    program::Instruction,
    rv32_auipc::solve_auipc,
};

const IMM_BITS: usize = 24;

type F = BabyBear;

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32AuipcChip<F>,
    rng: &mut StdRng,
    opcode: Rv32AuipcOpcode,
) {
    let imm = rng.gen_range(0..(1 << IMM_BITS));
    let a = gen_pointer(rng, 32);

    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize + Rv32AuipcOpcode::default_offset(),
            [a, 0, imm, 1, 0],
        ),
    );
    let initial_pc = tester
        .execution
        .records
        .last()
        .unwrap()
        .initial_state
        .pc
        .as_canonical_u32();

    let rd_data = solve_auipc(opcode, initial_pc, imm as u32);

    assert_eq!(rd_data.map(F::from_canonical_u32), tester.read::<4>(1, a));
}

#[test]
fn simple_execute_roundtrip_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32RdWriteAdapter::<F>::new();
    let inner = Rv32AuipcCoreChip::<F>::new(Rv32AuipcOpcode::default_offset());
    let mut chip = Rv32AuipcChip::<F>::new(adapter, inner, tester.memory_chip());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, AUIPC);
    }
}

#[test]
fn solve_auipc_sanity_test() {
    let opcode = AUIPC;
    let initial_pc = 234567890;
    let imm = 11302451;
    let rd_data = solve_auipc(opcode, initial_pc, imm);

    assert_eq!(rd_data, [210, 107, 113, 186]);
}
