use std::sync::Arc;

use derive_new::new;
use num_bigint::BigUint;
use openvm_circuit::arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilderConfig, FieldExpressionChipGPU, FieldExpressionCoreAir, FieldExpressionMetadata,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterExecutor};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{ec_double_ne_expr, EccRecord};

#[derive(new)]
pub struct WeierstrassDoubleChipGpu<const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub config: ExprBuilderConfig,
    pub offset: usize,
    pub a_biguint: BigUint,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for WeierstrassDoubleChipGpu<BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let range_bus = self.range_checker.cpu_chip.as_ref().unwrap().bus();
        let expr = ec_double_ne_expr(self.config.clone(), range_bus, self.a_biguint.clone());

        let total_input_limbs = expr.builder.num_input * expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            EccRecord<1, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;

        let local_opcode_idx = vec![
            Rv32WeierstrassOpcode::EC_DOUBLE as usize,
            Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
        ];

        let air = FieldExpressionCoreAir::new(expr, self.offset, local_opcode_idx, vec![]);

        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();

        let d_records = records.to_device().unwrap();

        let field_expr_chip = FieldExpressionChipGPU::new(
            air,
            d_records,
            num_records,
            record_size,
            adapter_width,
            BLOCKS,
            self.range_checker.clone(),
            self.bitwise_lookup.clone(),
            self.pointer_max_bits,
            self.timestamp_max_bits,
        );

        let d_trace = field_expr_chip.generate_field_trace();

        AirProvingContext::simple_no_pis(d_trace)
    }
}
