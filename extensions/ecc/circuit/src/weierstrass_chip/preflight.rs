//! Fast PreflightExecutor implementations for ECC operations that use native field arithmetic
//! instead of BigUint-based computation.

use openvm_algebra_circuit::fields::{get_field_type, FieldType};
use openvm_circuit::arch::{
    AdapterTraceExecutor, DynArray, ExecutionError, PreflightExecutor, RecordArena, VmStateMut,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpressionCoreRecordMut, FieldExpressionRecordLayout,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_circuit::system::memory::online::TracingMemory;
use strum::EnumCount;

use super::{
    add_ne::EcAddNeExecutor,
    curves::{ec_add_ne, ec_double, get_curve_type, CurveType},
    double::EcDoubleExecutor,
};

/// Compute EC point addition using fast native field arithmetic.
#[inline]
fn compute_ec_add_ne_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_type: Option<FieldType>,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> Option<[[u8; BLOCK_SIZE]; BLOCKS]> {
    let field_type = field_type?;

    Some(match field_type {
        FieldType::K256Coordinate => {
            ec_add_ne::<{ FieldType::K256Coordinate as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        FieldType::P256Coordinate => {
            ec_add_ne::<{ FieldType::P256Coordinate as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        FieldType::BN254Coordinate => {
            ec_add_ne::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        FieldType::BLS12_381Coordinate => {
            ec_add_ne::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        // Scalar fields are not used for ECC point coordinates
        _ => return None,
    })
}

/// Compute EC point doubling using fast native field arithmetic.
#[inline]
fn compute_ec_double_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    curve_type: Option<CurveType>,
    read_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> Option<[[u8; BLOCK_SIZE]; BLOCKS]> {
    let curve_type = curve_type?;

    Some(match curve_type {
        CurveType::K256 => {
            ec_double::<{ CurveType::K256 as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        CurveType::P256 => {
            ec_double::<{ CurveType::P256 as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        CurveType::BN254 => {
            ec_double::<{ CurveType::BN254 as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
        CurveType::BLS12_381 => {
            ec_double::<{ CurveType::BLS12_381 as u8 }, BLOCKS, BLOCK_SIZE>(read_data)
        }
    })
}

/// Check if this is a SETUP opcode (not a regular EC operation)
#[inline]
fn is_setup_opcode(local_opcode: usize) -> bool {
    let base_opcode = local_opcode % Rv32WeierstrassOpcode::COUNT;
    matches!(
        base_opcode,
        x if x == Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize
            || x == Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize
    )
}

// Implementation for EcAddNeExecutor
impl<F, RA, const BLOCKS: usize, const BLOCK_SIZE: usize> PreflightExecutor<F, RA>
    for EcAddNeExecutor<BLOCKS, BLOCK_SIZE>
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<
        'buf,
        FieldExpressionRecordLayout<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
        (
            <Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as AdapterTraceExecutor<F>>::RecordMut<'buf>,
            FieldExpressionCoreRecordMut<'buf>,
        ),
    >,
{
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, mut core_record) = state.ctx.alloc(self.0.get_record_layout());

        <Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as AdapterTraceExecutor<F>>::start(
            *state.pc,
            state.memory,
            &mut adapter_record,
        );

        let data: DynArray<u8> = self
            .0
            .adapter()
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let local_opcode = instruction.opcode.local_opcode_idx(self.0.offset);

        core_record.fill_from_execution_data(local_opcode as u8, &data.0);

        // Try fast path for non-SETUP operations with known field types
        let writes: DynArray<u8> = if !is_setup_opcode(local_opcode) {
            let field_type = get_field_type(&self.0.expr.prime);
            let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = data.clone().into();

            if let Some(output) = compute_ec_add_ne_fast::<BLOCKS, BLOCK_SIZE>(field_type, read_data)
            {
                output.into()
            } else {
                // Fall back to slow path - single operation chip, flag_idx = 0
                run_field_expression_precomputed::<true>(&self.0.expr, 0, &data.0)
            }
        } else {
            // SETUP operations: pass num_flags so no flag is set
            let no_flag = self.0.expr.num_flags();
            run_field_expression_precomputed::<true>(&self.0.expr, no_flag, &data.0)
        };

        self.0.adapter().write(
            state.memory,
            instruction,
            writes.into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.0.name.clone()
    }
}

// Implementation for EcDoubleExecutor
impl<F, RA, const BLOCKS: usize, const BLOCK_SIZE: usize> PreflightExecutor<F, RA>
    for EcDoubleExecutor<BLOCKS, BLOCK_SIZE>
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<
        'buf,
        FieldExpressionRecordLayout<
            F,
            Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
        (
            <Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as AdapterTraceExecutor<F>>::RecordMut<'buf>,
            FieldExpressionCoreRecordMut<'buf>,
        ),
    >,
{
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, mut core_record) = state.ctx.alloc(self.0.get_record_layout());

        <Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as AdapterTraceExecutor<F>>::start(
            *state.pc,
            state.memory,
            &mut adapter_record,
        );

        let data: DynArray<u8> = self
            .0
            .adapter()
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let local_opcode = instruction.opcode.local_opcode_idx(self.0.offset);

        core_record.fill_from_execution_data(local_opcode as u8, &data.0);

        // Try fast path for non-SETUP operations with known curve types
        let writes: DynArray<u8> = if !is_setup_opcode(local_opcode) {
            // Get the curve's a coefficient from setup_values
            let a_coeff = self.0.expr.setup_values.first();

            let curve_type = a_coeff.and_then(|a| get_curve_type(&self.0.expr.prime, a));
            let read_data: [[u8; BLOCK_SIZE]; BLOCKS] = data.clone().into();

            if let Some(output) = compute_ec_double_fast::<BLOCKS, BLOCK_SIZE>(curve_type, read_data)
            {
                output.into()
            } else {
                // Fall back to slow path - single operation chip, flag_idx = 0
                run_field_expression_precomputed::<true>(&self.0.expr, 0, &data.0)
            }
        } else {
            // SETUP operations: pass num_flags so no flag is set
            let no_flag = self.0.expr.num_flags();
            run_field_expression_precomputed::<true>(&self.0.expr, no_flag, &data.0)
        };

        self.0.adapter().write(
            state.memory,
            instruction,
            writes.into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.0.name.clone()
    }
}
