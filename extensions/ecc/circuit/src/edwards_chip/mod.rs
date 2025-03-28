mod add;
pub use add::*;

mod utils;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

use num_bigint::BigUint;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_ecc_transpiler::Rv32EdwardsOpcode;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionCoreChip};
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;
use utils::jacobi;

/// BLOCK_SIZE: how many cells do we read at a time, must be a power of 2.
/// BLOCKS: how many blocks do we need to represent one input or output
/// For example, for bls12_381, BLOCK_SIZE = 16, each element has 3 blocks and with two elements per input AffinePoint, BLOCKS = 6.
/// For secp256k1, BLOCK_SIZE = 32, BLOCKS = 2.
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct TeAddChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    TeAddChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        config: ExprBuilderConfig,
        offset: usize,
        a: BigUint,
        d: BigUint,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        // Ensure that the addition operation is complete
        assert!(jacobi(&a.clone().into(), &config.modulus.clone().into()) == 1);
        assert!(jacobi(&d.clone().into(), &config.modulus.clone().into()) == -1);

        let expr = ec_add_expr(config, range_checker.bus(), a, d);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Rv32EdwardsOpcode::TE_ADD as usize,
                Rv32EdwardsOpcode::SETUP_TE_ADD as usize,
            ],
            vec![],
            range_checker,
            "TeEcAdd",
            true,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}
