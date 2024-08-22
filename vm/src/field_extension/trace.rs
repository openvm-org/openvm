use std::{array, vec::IntoIter};

use itertools::Itertools;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use afs_stark_backend::rap::AnyRap;

use crate::{
    arch::{chips::MachineChip, instructions::OpCode},
    memory::offline_checker::columns::MemoryOfflineCheckerAuxCols,
};

use super::{
    columns::{
        FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols,
        FieldExtensionArithmeticIoCols,
    },
    EXTENSION_DEGREE, FieldExtensionArithmetic, FieldExtensionArithmeticChip,
    FieldExtensionArithmeticOperation,
};

/// Constructs a new set of columns (including auxiliary columns) given inputs.
fn generate_cols<F: Field>(
    op: FieldExtensionArithmeticOperation<F>,
    oc_aux_iter: &mut IntoIter<MemoryOfflineCheckerAuxCols<1, F>>,
) -> FieldExtensionArithmeticCols<F> {
    let is_add = F::from_bool(op.opcode == OpCode::FE4ADD);
    let is_sub = F::from_bool(op.opcode == OpCode::FE4SUB);
    let is_mul = F::from_bool(op.opcode == OpCode::BBE4MUL);
    let is_inv = F::from_bool(op.opcode == OpCode::BBE4INV);

    let x = op.operand1;
    let y = op.operand2;

    let inv = if x == [F::zero(); EXTENSION_DEGREE] {
        [F::zero(); EXTENSION_DEGREE]
    } else {
        FieldExtensionArithmetic::solve(OpCode::BBE4INV, x, y).unwrap()
    };

    FieldExtensionArithmeticCols {
        io: FieldExtensionArithmeticIoCols {
            opcode: F::from_canonical_usize(op.opcode as usize),
            pc: F::from_canonical_usize(op.pc),
            timestamp: F::from_canonical_usize(op.start_timestamp),
            x,
            y,
            z: op.result,
        },
        aux: FieldExtensionArithmeticAuxCols {
            is_valid: F::one(),
            valid_y_read: F::one() - is_inv,
            op_a: op.op_a,
            op_b: op.op_b,
            op_c: op.op_c,
            d: op.d,
            e: op.e,
            is_add,
            is_sub,
            is_mul,
            is_inv,
            inv,
            mem_oc_aux_cols: array::from_fn(|_| oc_aux_iter.next().unwrap()),
        },
    }
}

impl<F: PrimeField32> MachineChip<F> for FieldExtensionArithmeticChip<F> {
    /// Generates trace for field arithmetic chip.
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        // todo[zach]: it's weird that `generate_trace` mutates the receiver
        let accesses = self.memory.take_accesses_buffer();
        let mut accesses_iter = accesses.into_iter();

        let mut trace: Vec<F> = self
            .operations
            .iter()
            .cloned()
            .flat_map(|op| generate_cols(op, &mut accesses_iter).flatten())
            .collect();

        assert!(accesses_iter.next().is_none());

        let curr_height = self.operations.len();
        let correct_height = curr_height.next_power_of_two();

        let width = FieldExtensionArithmeticCols::<F>::get_width(&self.air);
        trace.extend(
            (0..correct_height - curr_height)
                .flat_map(|_| self.make_blank_row().flatten())
                .collect_vec(),
        );

        RowMajorMatrix::new(trace, width)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.operations.len()
    }

    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
