use axvm_instructions::Rv32BaseAlu256Opcode;
use p3_baby_bear::BabyBear;
use strum_macros::EnumIter;

use crate::{
    arch::{
        extensions::{EXECUTION_BUS, PROGRAM_BUS},
        SystemBase, SystemComplex, SystemConfig, VmExtensionBuilder, VmInventory,
    },
    rv32im::{adapters::Rv32BaseAluAdapterChip, Rv32BaseAluChip},
};

type F = BabyBear;

#[test]
fn test_rv32i() {
    let system_complex = SystemComplex::<F>::new(SystemConfig::default());
    let builder = system_complex.extension_builder();

    let mut rv32i = VmInventory::new();
    rv32i.add_executor(
        Rv32BaseAluChip::<F>::new(
            Rv32BaseAluAdapterChip::new(EXECUTION_BUS, PROGRAM_BUS, builder.memory_controller()),
            BaseAluCoreChip::new(),
            builder.memory_controller(),
        ),
        Rv32BaseAlu256Opcode::iter(),
    );

    todo!()
}
