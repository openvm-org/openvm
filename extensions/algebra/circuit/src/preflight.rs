//! Fast PreflightExecutor implementations that use native field arithmetic
//! instead of BigUint-based computation.
//!
//! During preflight execution, we only need to compute outputs to write to memory.
//! The trace filler will re-execute with BigUint arithmetic for constraint generation.
//! This module optimizes the preflight path by using native field arithmetic for known
//! field types (K256, P256, BN254, BLS12-381).

use openvm_algebra_transpiler::{Fp2Opcode, Rv32ModularArithmeticOpcode};
use openvm_circuit::{
    arch::{ExecutionError, PreflightExecutor, RecordArena, VmStateMut},
    system::memory::online::TracingMemory,
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpressionCoreRecordMut, FieldExpressionRecordLayout,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;
use openvm_stark_backend::p3_field::PrimeField32;
use strum::EnumCount;

use crate::{
    fields::{field_operation, fp2_operation, FieldType, Operation},
    FieldExprVecHeapExecutor,
};

/// Generates a match statement dispatching (FieldType, Operation) pairs to a const-generic
/// function call. Mirrors the `generate_field_dispatch!` macro in `execution.rs`.
macro_rules! dispatch_field_op {
    // Exhaustive variant (all field type × operation combinations listed)
    ($fn:ident, $field_type:expr, $op:expr, $read_data:expr,
     [$(($curve:ident, $operation:ident)),* $(,)?]) => {
        match ($field_type, $op) {
            $(
                (FieldType::$curve, Operation::$operation) => $fn::<
                    { FieldType::$curve as u8 },
                    BLOCKS,
                    BLOCK_SIZE,
                    { Operation::$operation as u8 },
                >($read_data),
            )*
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    };
}

/// Compute output using fast native field arithmetic for known field types.
/// Returns None if the field type is not supported (falls back to slow path).
#[inline]
fn compute_output_fast<const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>(
    field_type: Option<FieldType>,
    operation: Option<Operation>,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> Option<[[u8; BLOCK_SIZE]; BLOCKS]> {
    let field_type = field_type?;
    let op = operation?;

    if IS_FP2 {
        Some(compute_fp2_fast::<BLOCKS, BLOCK_SIZE>(
            field_type, op, read_data,
        ))
    } else {
        Some(compute_field_fast::<BLOCKS, BLOCK_SIZE>(
            field_type, op, read_data,
        ))
    }
}

/// Compute field operation using native arithmetic.
#[inline]
fn compute_field_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_type: FieldType,
    op: Operation,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    dispatch_field_op!(
        field_operation,
        field_type,
        op,
        read_data,
        [
            (K256Coordinate, Add),
            (K256Coordinate, Sub),
            (K256Coordinate, Mul),
            (K256Coordinate, Div),
            (K256Scalar, Add),
            (K256Scalar, Sub),
            (K256Scalar, Mul),
            (K256Scalar, Div),
            (P256Coordinate, Add),
            (P256Coordinate, Sub),
            (P256Coordinate, Mul),
            (P256Coordinate, Div),
            (P256Scalar, Add),
            (P256Scalar, Sub),
            (P256Scalar, Mul),
            (P256Scalar, Div),
            (BN254Coordinate, Add),
            (BN254Coordinate, Sub),
            (BN254Coordinate, Mul),
            (BN254Coordinate, Div),
            (BN254Scalar, Add),
            (BN254Scalar, Sub),
            (BN254Scalar, Mul),
            (BN254Scalar, Div),
            (BLS12_381Coordinate, Add),
            (BLS12_381Coordinate, Sub),
            (BLS12_381Coordinate, Mul),
            (BLS12_381Coordinate, Div),
            (BLS12_381Scalar, Add),
            (BLS12_381Scalar, Sub),
            (BLS12_381Scalar, Mul),
            (BLS12_381Scalar, Div),
        ]
    )
}

/// Compute Fp2 operation using native arithmetic.
#[inline]
fn compute_fp2_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_type: FieldType,
    op: Operation,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    dispatch_field_op!(
        fp2_operation,
        field_type,
        op,
        read_data,
        [
            (BN254Coordinate, Add),
            (BN254Coordinate, Sub),
            (BN254Coordinate, Mul),
            (BN254Coordinate, Div),
            (BLS12_381Coordinate, Add),
            (BLS12_381Coordinate, Sub),
            (BLS12_381Coordinate, Mul),
            (BLS12_381Coordinate, Div),
        ]
    )
}

/// Convert local opcode to Operation enum for modular arithmetic.
#[inline]
fn local_opcode_to_modular_operation(local_opcode: usize) -> Option<Operation> {
    let base_opcode = local_opcode % Rv32ModularArithmeticOpcode::COUNT;
    match base_opcode {
        x if x == Rv32ModularArithmeticOpcode::ADD as usize => Some(Operation::Add),
        x if x == Rv32ModularArithmeticOpcode::SUB as usize => Some(Operation::Sub),
        x if x == Rv32ModularArithmeticOpcode::MUL as usize => Some(Operation::Mul),
        x if x == Rv32ModularArithmeticOpcode::DIV as usize => Some(Operation::Div),
        _ => None,
    }
}

/// Convert local opcode to Operation enum for Fp2 arithmetic.
#[inline]
fn local_opcode_to_fp2_operation(local_opcode: usize) -> Option<Operation> {
    let base_opcode = local_opcode % Fp2Opcode::COUNT;
    match base_opcode {
        x if x == Fp2Opcode::ADD as usize => Some(Operation::Add),
        x if x == Fp2Opcode::SUB as usize => Some(Operation::Sub),
        x if x == Fp2Opcode::MUL as usize => Some(Operation::Mul),
        x if x == Fp2Opcode::DIV as usize => Some(Operation::Div),
        _ => None,
    }
}

impl<F, RA, const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    PreflightExecutor<F, RA> for FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<
        'buf,
        FieldExpressionRecordLayout<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
        (
            <Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> as openvm_circuit::arch::AdapterTraceExecutor<F>>::RecordMut<'buf>,
            FieldExpressionCoreRecordMut<'buf>,
        ),
    >,
{
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        use openvm_circuit::arch::AdapterTraceExecutor;

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

        core_record
            .fill_from_execution_data(local_opcode as u8, read_data.as_flattened().as_flattened());

        // Try fast path: use native field arithmetic for known field types and non-setup operations
        let operation = if IS_FP2 {
            local_opcode_to_fp2_operation(local_opcode)
        } else {
            local_opcode_to_modular_operation(local_opcode)
        };

        let output: [[u8; BLOCK_SIZE]; BLOCKS] =
            if let Some(output) = compute_output_fast::<BLOCKS, BLOCK_SIZE, IS_FP2>(
                self.cached_field_type,
                operation,
                read_data,
            ) {
                output
            } else {
                // Fall back to slow path for unsupported field types or SETUP operations
                let flag_idx = self
                    .inner
                    .local_opcode_idx
                    .iter()
                    .position(|&idx| idx == local_opcode)
                    .and_then(|pos| {
                        if pos < self.inner.opcode_flag_idx.len() {
                            Some(self.inner.opcode_flag_idx[pos])
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| self.inner.expr.num_flags());

                run_field_expression_precomputed::<true>(
                    &self.inner.expr,
                    flag_idx,
                    read_data.as_flattened().as_flattened(),
                )
                .into()
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
