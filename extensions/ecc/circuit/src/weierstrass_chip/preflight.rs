//! Fast PreflightExecutor implementations for ECC operations that use native field arithmetic
//! instead of BigUint-based computation.

use openvm_algebra_circuit::fields::FieldType;
use openvm_circuit::{
    arch::{AdapterTraceExecutor, ExecutionError, PreflightExecutor, RecordArena, VmStateMut},
    system::memory::online::TracingMemory,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpressionCoreRecordMut, FieldExpressionRecordLayout,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;
use openvm_stark_backend::p3_field::PrimeField32;
use strum::EnumCount;

use super::{
    add_ne::EcAddNeExecutor,
    curves::{ec_add_ne, ec_double, CurveType},
    double::EcDoubleExecutor,
};

/// Generates a match statement dispatching an enum to const-generic function calls.
macro_rules! dispatch_enum {
    ($fn:ident, $val:expr, $read_data:expr,
     [$(($variant_type:ident :: $variant:ident)),* $(,)?]) => {
        match $val {
            $(
                $variant_type::$variant => $fn::<
                    { $variant_type::$variant as u8 },
                    BLOCKS,
                    BLOCK_SIZE,
                >($read_data),
            )*
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    };
}

/// Compute EC point addition using fast native field arithmetic.
#[inline]
fn compute_ec_add_ne_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_type: Option<FieldType>,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> Option<[[u8; BLOCK_SIZE]; BLOCKS]> {
    let field_type = field_type?;

    Some(match field_type {
        FieldType::K256Coordinate
        | FieldType::P256Coordinate
        | FieldType::BN254Coordinate
        | FieldType::BLS12_381Coordinate => {
            dispatch_enum!(
                ec_add_ne,
                field_type,
                read_data,
                [
                    (FieldType::K256Coordinate),
                    (FieldType::P256Coordinate),
                    (FieldType::BN254Coordinate),
                    (FieldType::BLS12_381Coordinate),
                ]
            )
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

    Some(dispatch_enum!(
        ec_double,
        curve_type,
        read_data,
        [
            (CurveType::K256),
            (CurveType::P256),
            (CurveType::BN254),
            (CurveType::BLS12_381),
        ]
    ))
}

/// Check if this is a SETUP opcode (not a regular EC operation)
#[inline]
fn is_setup_opcode(local_opcode: usize) -> bool {
    let base_opcode = local_opcode % Rv32WeierstrassOpcode::COUNT;
    base_opcode == Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize
        || base_opcode == Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize
}

/// Slow-path fallback for EC operations (used for SETUP opcodes and unknown field types).
#[inline]
fn compute_ec_slow<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    expr: &openvm_mod_circuit_builder::FieldExpr,
    local_opcode: usize,
    read_bytes: &[u8],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let flag_idx = if is_setup_opcode(local_opcode) {
        // SETUP operations: pass num_flags so no flag is set
        expr.num_flags()
    } else {
        // Single operation chip, flag_idx = 0
        0
    };
    run_field_expression_precomputed::<true>(expr, flag_idx, read_bytes).into()
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
        let (mut adapter_record, mut core_record) =
            state.ctx.alloc(self.inner.get_record_layout());

        <Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as AdapterTraceExecutor<F>>::start(
            *state.pc,
            state.memory,
            &mut adapter_record,
        );

        let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = self
            .inner
            .adapter()
            .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = instruction.opcode.local_opcode_idx(self.inner.offset);

        core_record.fill_from_execution_data(
            local_opcode as u8,
            read_data.as_flattened().as_flattened(),
        );

        // Try fast path for non-SETUP operations with known field types
        let output: [[u8; BLOCK_SIZE]; BLOCKS] =
            if !is_setup_opcode(local_opcode) {
                compute_ec_add_ne_fast::<BLOCKS, BLOCK_SIZE>(self.cached_field_type, read_data)
                    .unwrap_or_else(|| {
                        compute_ec_slow::<BLOCKS, BLOCK_SIZE>(
                            &self.inner.expr,
                            local_opcode,
                            read_data.as_flattened().as_flattened(),
                        )
                    })
            } else {
                compute_ec_slow::<BLOCKS, BLOCK_SIZE>(
                    &self.inner.expr,
                    local_opcode,
                    read_data.as_flattened().as_flattened(),
                )
            };

        self.inner.adapter().write(
            state.memory,
            instruction,
            output,
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.inner.name.clone()
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
        let (mut adapter_record, mut core_record) =
            state.ctx.alloc(self.inner.get_record_layout());

        <Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as AdapterTraceExecutor<F>>::start(
            *state.pc,
            state.memory,
            &mut adapter_record,
        );

        let read_data_arr: [[[u8; BLOCK_SIZE]; BLOCKS]; 1] = self
            .inner
            .adapter()
            .read(state.memory, instruction, &mut adapter_record);
        let read_data: [[u8; BLOCK_SIZE]; BLOCKS] = read_data_arr[0];

        let local_opcode = instruction.opcode.local_opcode_idx(self.inner.offset);

        core_record.fill_from_execution_data(local_opcode as u8, read_data.as_flattened());

        // Try fast path for non-SETUP operations with known curve types
        let output: [[u8; BLOCK_SIZE]; BLOCKS] =
            if !is_setup_opcode(local_opcode) {
                compute_ec_double_fast::<BLOCKS, BLOCK_SIZE>(self.cached_curve_type, read_data)
                    .unwrap_or_else(|| {
                        compute_ec_slow::<BLOCKS, BLOCK_SIZE>(
                            &self.inner.expr,
                            local_opcode,
                            read_data.as_flattened(),
                        )
                    })
            } else {
                compute_ec_slow::<BLOCKS, BLOCK_SIZE>(
                    &self.inner.expr,
                    local_opcode,
                    read_data.as_flattened(),
                )
            };

        self.inner.adapter().write(
            state.memory,
            instruction,
            output,
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.inner.name.clone()
    }
}
