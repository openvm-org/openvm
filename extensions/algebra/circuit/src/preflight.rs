//! Fast PreflightExecutor implementations that use native field arithmetic
//! instead of BigUint-based computation.
//!
//! During preflight execution, we only need to compute outputs to write to memory.
//! The trace filler will re-execute with BigUint arithmetic for constraint generation.
//! This module optimizes the preflight path by using native field arithmetic for known
//! field types (K256, P256, BN254, BLS12-381).

use openvm_algebra_transpiler::{Fp2Opcode, Rv32ModularArithmeticOpcode};
use openvm_circuit::arch::{
    DynArray, ExecutionError, PreflightExecutor, RecordArena, VmStateMut,
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpressionCoreRecordMut, FieldExpressionRecordLayout,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_circuit::system::memory::online::TracingMemory;

use crate::{
    fields::{
        field_operation, fp2_operation, get_field_type, get_fp2_field_type, FieldType, Operation,
    },
    FieldExprVecHeapExecutor,
};

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
        Some(compute_fp2_fast::<BLOCKS, BLOCK_SIZE>(field_type, op, read_data))
    } else {
        Some(compute_field_fast::<BLOCKS, BLOCK_SIZE>(field_type, op, read_data))
    }
}

/// Compute field operation using native arithmetic.
/// Dispatches to the appropriate field type and operation at runtime.
#[inline]
fn compute_field_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_type: FieldType,
    op: Operation,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match (field_type, op) {
        (FieldType::K256Coordinate, Operation::Add) => {
            field_operation::<{ FieldType::K256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::K256Coordinate, Operation::Sub) => {
            field_operation::<{ FieldType::K256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::K256Coordinate, Operation::Mul) => {
            field_operation::<{ FieldType::K256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::K256Coordinate, Operation::Div) => {
            field_operation::<{ FieldType::K256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::K256Scalar, Operation::Add) => {
            field_operation::<{ FieldType::K256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::K256Scalar, Operation::Sub) => {
            field_operation::<{ FieldType::K256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::K256Scalar, Operation::Mul) => {
            field_operation::<{ FieldType::K256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::K256Scalar, Operation::Div) => {
            field_operation::<{ FieldType::K256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::P256Coordinate, Operation::Add) => {
            field_operation::<{ FieldType::P256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::P256Coordinate, Operation::Sub) => {
            field_operation::<{ FieldType::P256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::P256Coordinate, Operation::Mul) => {
            field_operation::<{ FieldType::P256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::P256Coordinate, Operation::Div) => {
            field_operation::<{ FieldType::P256Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::P256Scalar, Operation::Add) => {
            field_operation::<{ FieldType::P256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::P256Scalar, Operation::Sub) => {
            field_operation::<{ FieldType::P256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::P256Scalar, Operation::Mul) => {
            field_operation::<{ FieldType::P256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::P256Scalar, Operation::Div) => {
            field_operation::<{ FieldType::P256Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Add) => {
            field_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Sub) => {
            field_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Mul) => {
            field_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Div) => {
            field_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::BN254Scalar, Operation::Add) => {
            field_operation::<{ FieldType::BN254Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::BN254Scalar, Operation::Sub) => {
            field_operation::<{ FieldType::BN254Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::BN254Scalar, Operation::Mul) => {
            field_operation::<{ FieldType::BN254Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::BN254Scalar, Operation::Div) => {
            field_operation::<{ FieldType::BN254Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Add) => {
            field_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Sub) => {
            field_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Mul) => {
            field_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Div) => {
            field_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::BLS12_381Scalar, Operation::Add) => {
            field_operation::<{ FieldType::BLS12_381Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::BLS12_381Scalar, Operation::Sub) => {
            field_operation::<{ FieldType::BLS12_381Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::BLS12_381Scalar, Operation::Mul) => {
            field_operation::<{ FieldType::BLS12_381Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::BLS12_381Scalar, Operation::Div) => {
            field_operation::<{ FieldType::BLS12_381Scalar as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
    }
}

/// Compute Fp2 operation using native arithmetic.
#[inline]
fn compute_fp2_fast<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_type: FieldType,
    op: Operation,
    read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match (field_type, op) {
        (FieldType::BN254Coordinate, Operation::Add) => {
            fp2_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Sub) => {
            fp2_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Mul) => {
            fp2_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::BN254Coordinate, Operation::Div) => {
            fp2_operation::<{ FieldType::BN254Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Add) => {
            fp2_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Add as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Sub) => {
            fp2_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Sub as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Mul) => {
            fp2_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Mul as u8 }>(read_data)
        }
        (FieldType::BLS12_381Coordinate, Operation::Div) => {
            fp2_operation::<{ FieldType::BLS12_381Coordinate as u8 }, BLOCKS, BLOCK_SIZE, { Operation::Div as u8 }>(read_data)
        }
        _ => panic!("Unsupported field type for Fp2: {:?}", field_type),
    }
}

/// Convert local opcode to Operation enum for modular arithmetic.
#[inline]
fn local_opcode_to_modular_operation(local_opcode: usize) -> Option<Operation> {
    // The local opcode within a modulus offset maps to Rv32ModularArithmeticOpcode
    let base_opcode = local_opcode % Rv32ModularArithmeticOpcode::COUNT;
    match base_opcode {
        x if x == Rv32ModularArithmeticOpcode::ADD as usize => Some(Operation::Add),
        x if x == Rv32ModularArithmeticOpcode::SUB as usize => Some(Operation::Sub),
        x if x == Rv32ModularArithmeticOpcode::MUL as usize => Some(Operation::Mul),
        x if x == Rv32ModularArithmeticOpcode::DIV as usize => Some(Operation::Div),
        // SETUP operations return None - they need the slow path
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
        // SETUP operations return None - they need the slow path
        _ => None,
    }
}

use strum::EnumCount;

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

        // Try fast path: use native field arithmetic for known field types and non-setup operations
        let writes: DynArray<u8> = {
            let field_type = if IS_FP2 {
                get_fp2_field_type(&self.0.expr.prime)
            } else {
                get_field_type(&self.0.expr.prime)
            };

            let operation = if IS_FP2 {
                local_opcode_to_fp2_operation(local_opcode)
            } else {
                local_opcode_to_modular_operation(local_opcode)
            };

            // Convert data to the expected array format for fast path
            let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = data.clone().into();

            if let Some(output) = compute_output_fast::<BLOCKS, BLOCK_SIZE, IS_FP2>(
                field_type,
                operation,
                read_data,
            ) {
                output.into()
            } else {
                // Fall back to slow path for unsupported field types or SETUP operations
                let flag_idx = self
                    .0
                    .local_opcode_idx
                    .iter()
                    .position(|&idx| idx == local_opcode)
                    .and_then(|pos| {
                        if pos < self.0.opcode_flag_idx.len() {
                            Some(self.0.opcode_flag_idx[pos])
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| self.0.expr.num_flags());

                run_field_expression_precomputed::<true>(&self.0.expr, flag_idx, &data.0)
            }
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
