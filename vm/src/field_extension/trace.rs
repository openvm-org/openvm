use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;

use afs_stark_backend::rap::AnyRap;

use crate::arch::{chips::MachineChip, instructions::OpCode};

use super::{
    columns::{
        FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols,
        FieldExtensionArithmeticIoCols,
    },
    FieldExtensionArithmeticAir, FieldExtensionArithmeticChip, FieldExtensionArithmeticOperation,
};

/// Constructs a new set of columns (including auxiliary columns) given inputs.
fn generate_cols<T: Field>(
    op: FieldExtensionArithmeticOperation<T>,
) -> FieldExtensionArithmeticCols<T> {
    let opcode_value = op.opcode as u32 - FieldExtensionArithmeticAir::BASE_OP as u32;
    let opcode_lo_u32 = opcode_value % 2;
    let opcode_hi_u32 = opcode_value / 2;
    let opcode_lo = T::from_canonical_u32(opcode_lo_u32);
    let opcode_hi = T::from_canonical_u32(opcode_hi_u32);
    let is_mul = T::from_bool(op.opcode == OpCode::BBE4MUL);
    let is_inv = T::from_bool(op.opcode == OpCode::BBE4INV);

    let x = op.operand1;
    let y = op.operand2;

    let add_sub_coeff = T::one() - T::two() * opcode_lo;

    let sum_or_diff = [
        x[0] + add_sub_coeff * y[0],
        x[1] + add_sub_coeff * y[1],
        x[2] + add_sub_coeff * y[2],
        x[3] + add_sub_coeff * y[3],
    ];
    let product = FieldExtensionArithmeticAir::solve(OpCode::BBE4MUL, x, y).unwrap();
    let inv = if x[0] == T::zero() && x[1] == T::zero() && x[2] == T::zero() && x[3] == T::zero() {
        [T::zero(), T::zero(), T::zero(), T::zero()]
    } else {
        FieldExtensionArithmeticAir::solve(OpCode::BBE4INV, x, y).unwrap()
    };

    FieldExtensionArithmeticCols {
        io: FieldExtensionArithmeticIoCols {
            opcode: T::from_canonical_usize(op.opcode as usize),
            x,
            y,
            z: op.result,
        },
        aux: FieldExtensionArithmeticAuxCols {
            is_valid: T::one(),
            valid_y_read: T::one() - is_inv,
            pc: T::from_canonical_usize(op.pc),
            start_timestamp: T::from_canonical_usize(op.start_timestamp),
            op_a: op.op_a,
            op_b: op.op_b,
            op_c: op.op_c,
            d: op.d,
            e: op.e,
            opcode_lo,
            opcode_hi,
            is_mul,
            is_inv,
            sum_or_diff,
            product,
            inv,
        },
    }
}

impl<'a, F: PrimeField32> MachineChip<'a, F> for FieldExtensionArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let mut trace: Vec<F> = self
            .operations
            .iter()
            .flat_map(|op| generate_cols(*op).flatten())
            .collect();

        let empty_row: Vec<F> = FieldExtensionArithmeticCols::blank_row().flatten();
        let curr_height = self.operations.len();
        let correct_height = curr_height.next_power_of_two();
        trace.extend(
            empty_row.iter().cloned().cycle().take(
                (correct_height - curr_height) * FieldExtensionArithmeticCols::<F>::get_width(),
            ),
        );

        RowMajorMatrix::new(trace, FieldExtensionArithmeticCols::<F>::get_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC> {
        &self.air
    }

    fn current_trace_height(&self) -> usize {
        self.operations.len()
    }
}
