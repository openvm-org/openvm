use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use axvm_circuit::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmExtension, VmGenericConfig,
        VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    extensions::{
        rv32_io::{Rv32HintStore, Rv32HintStoreExecutor},
        rv32im::{Rv32I, Rv32M},
    },
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::*;
use derive_more::derive::From;
use p3_field::PrimeField32;
use program::DEFAULT_PC_STEP;
use strum::IntoEnumIterator;

use super::WeierstrassExtension;

pub struct Rv32WeierstrassConfig {
    pub system: SystemConfig,
    pub base: Rv32I,
    pub mul: Rv32M,
    pub io: Rv32HintStore,

    pub modular: ModularExtension,
    pub weierstrass: WeierstrassExtension,
}
