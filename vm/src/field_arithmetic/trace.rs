use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use afs_stark_backend::rap::AnyRap;

use crate::{arch::chips::MachineChip, cpu::OpCode};

use super::{
    ArithmeticOperation,
    columns::{FieldArithmeticAuxCols, FieldArithmeticCols, FieldArithmeticIoCols}, FieldArithmeticAir, FieldArithmeticChip,
};

/// Constructs a new set of columns (including auxiliary columns) given inputs.
fn generate_cols<T: PrimeField32>(operation: &ArithmeticOperation<T>) -> FieldArithmeticCols<T> {
    let opcode_u32 = operation.instruction.opcode.as_canonical_u32();
    let opcode = OpCode::from_u8(opcode_u32 as u8).unwrap();
    let opcode_value = opcode_u32 - FieldArithmeticAir::BASE_OP as u32;
    let opcode_lo_u32 = opcode_value % 2;
    let opcode_hi_u32 = opcode_value / 2;
    let opcode_lo = T::from_canonical_u32(opcode_lo_u32);
    let opcode_hi = T::from_canonical_u32(opcode_hi_u32);
    let is_div = T::from_bool(opcode == OpCode::FDIV);
    let is_mul = T::from_bool(opcode == OpCode::FMUL);

    let x = operation.operand1;
    let y = operation.operand2;
    let sum_or_diff = x + y - T::two() * opcode_lo * y;
    let product = x * y;

    let divisor_inv = if opcode == OpCode::FDIV {
        y.inverse()
    } else {
        T::zero()
    };
    let quotient = x * divisor_inv;
    let z = operation.result;

    let instruction = operation.instruction;

    FieldArithmeticCols {
        io: FieldArithmeticIoCols {
            rcv_count: T::one(),
            opcode: instruction.opcode,
            z_address: instruction.a,
            x_address: instruction.b,
            y_address: instruction.c,
            xz_as: instruction.d,
            y_as: instruction.e,
            x,
            y,
            z,
            prev_state: operation.prev_state.map(T::from_canonical_usize),
        },
        aux: FieldArithmeticAuxCols {
            opcode_lo,
            opcode_hi,
            is_mul,
            is_div,
            sum_or_diff,
            product,
            quotient,
            divisor_inv,
        },
    }
}

impl<F: PrimeField32> MachineChip<F> for FieldArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let mut trace: Vec<F> = self
            .operations
            .iter()
            .flat_map(|operation| generate_cols(operation).flatten())
            .collect();

        let empty_row: Vec<F> = FieldArithmeticCols::blank_row().flatten();
        let curr_height = self.operations.len();
        let correct_height = curr_height.next_power_of_two();
        trace.extend(
            empty_row
                .iter()
                .cloned()
                .cycle()
                .take((correct_height - curr_height) * FieldArithmeticCols::<F>::get_width()),
        );

        RowMajorMatrix::new(trace, FieldArithmeticCols::<F>::get_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        &self.air
    }
}
