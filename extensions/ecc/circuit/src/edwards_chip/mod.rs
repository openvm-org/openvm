mod add;
pub use add::*;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_circuit_primitives_derive::{BytesStateful, Chip, ChipUsageGetter};
use openvm_ecc_transpiler::Rv32EdwardsOpcode;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionCoreChip};
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

/// BLOCK_SIZE: how many cells do we read at a time, must be a power of 2.
/// BLOCKS: how many blocks do we need to represent one input or output
/// For example, for bls12_381, BLOCK_SIZE = 16, each element has 3 blocks and with two elements per input AffinePoint, BLOCKS = 6.
/// For secp256k1, BLOCK_SIZE = 32, BLOCKS = 2.
#[derive(Chip, ChipUsageGetter, InstructionExecutor, BytesStateful)]
pub struct TeEcAddChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

// converts from num_bigint::BigUint to num_bigint_dig::BigInt in order to use num_bigint_dig::algorithms::jacobi
fn num_bigint_to_num_bigint_dig(x: &num_bigint::BigUint) -> num_bigint_dig::BigInt {
    num_bigint_dig::BigInt::from_bytes_le(num_bigint_dig::Sign::Plus, &x.to_bytes_le())
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    TeEcAddChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        config: ExprBuilderConfig,
        offset: usize,
        a: num_bigint::BigUint,
        d: num_bigint::BigUint,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        // Ensure that the addition operation is complete
        assert!(
            num_bigint_dig::algorithms::jacobi(
                &num_bigint_to_num_bigint_dig(&a),
                &num_bigint_to_num_bigint_dig(&config.modulus)
            ) == 1
        );
        assert!(
            num_bigint_dig::algorithms::jacobi(
                &num_bigint_to_num_bigint_dig(&d),
                &num_bigint_to_num_bigint_dig(&config.modulus)
            ) == -1
        );

        let expr = ec_add_expr(config, range_checker.bus(), a, d);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Rv32EdwardsOpcode::EC_ADD as usize,
                Rv32EdwardsOpcode::SETUP_EC_ADD as usize,
            ],
            vec![],
            range_checker,
            "TeEcAdd",
            true,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}
