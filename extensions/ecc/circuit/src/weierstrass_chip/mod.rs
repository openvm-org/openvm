mod add_ne;
mod double;

pub use add_ne::*;
pub use double::*;

#[cfg(test)]
mod tests;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionCoreChip};
use axvm_circuit::{arch::VmChipWrapper, system::memory::MemoryControllerRef};
use axvm_circuit_derive::InstructionExecutor;
use axvm_ecc_transpiler::Rv32WeierstrassOpcode;
use axvm_rv32_adapters::Rv32VecHeapAdapterChip;
use num_bigint_dig::BigUint;
use ax_stark_backend::p3_field::PrimeField32;

/// BLOCK_SIZE: how many cells do we read at a time, must be a power of 2.
/// BLOCKS: how many blocks do we need to represent one input or output
/// For example, for bls12_381, BLOCK_SIZE = 16, each element has 3 blocks and with two elements per input AffinePoint, BLOCKS = 6.
/// For secp256k1, BLOCK_SIZE = 32, BLOCKS = 2.
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcAddNeChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    EcAddNeChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
    ) -> Self {
        let expr = ec_add_ne_expr(config, memory_controller.borrow().range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Rv32WeierstrassOpcode::EC_ADD_NE as usize,
                Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
            ],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "EcAddNe",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcDoubleChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    EcDoubleChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        offset: usize,
        a: BigUint,
    ) -> Self {
        let expr = ec_double_expr(config, memory_controller.borrow().range_checker.bus(), a);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Rv32WeierstrassOpcode::EC_DOUBLE as usize,
                Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
            ],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "EcDouble",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}
