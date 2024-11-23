use axvm_instructions::Rv32BaseAlu256Opcode;
use p3_baby_bear::BabyBear;
use strum_macros::EnumIter;

use crate::{
    arch::{SystemBase, SystemComplex, SystemConfig, VmInventory, VmInventoryBuilder},
    rv32im::{adapters::Rv32BaseAluAdapterChip, Rv32BaseAluChip},
};

type F = BabyBear;

#[test]
fn test_rv32i() {
    let system_complex = SystemComplex::<F>::new(SystemConfig::default());
    let builder = system_complex.inventory_builder();
    let execution_bus = builder.system_base().execution_bus();
    let program_bus = builder.system_base().program_bus();

    // let mut rv32i = VmInventory::new();
    // rv32i.add_executor(
    //     Rv32BaseAluChip::<F>::new(
    //         Rv32BaseAluAdapterChip::new(
    //             execution_bus,
    //             program_bus,
    //             builder.memory_controller().clone(),
    //         ),
    //         BaseAluCoreChip::new(),
    //         builder.memory_controller().clone(),
    //     ),
    //     Rv32BaseAlu256Opcode::iter(),
    // );

    todo!()
}
