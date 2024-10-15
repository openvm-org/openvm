use std::borrow::{Borrow, BorrowMut};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use itertools::izip;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            FieldArithmeticOpcode,
            FieldArithmeticOpcode::{ADD, DIV, MUL, SUB},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NewFieldArithmeticCols<T> {
    pub a: T,
    pub b: T,
    pub c: T,

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,
}

#[derive(Copy, Clone, Debug)]
pub struct NewFieldArithmeticCoreAir {
    offset: usize,
}

impl<F: Field> BaseAir<F> for NewFieldArithmeticCoreAir {
    fn width(&self) -> usize {
        NewFieldArithmeticCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for NewFieldArithmeticCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for NewFieldArithmeticCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 2]>,
    I::Writes: From<[[AB::Expr; 1]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &NewFieldArithmeticCols<_> = local_core.borrow();

        let a = cols.a;
        let b = cols.b;
        let c = cols.c;

        let flags = [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div];
        let opcodes = [ADD, SUB, MUL, DIV];
        let results = [b + c, b - c, b * c, b * cols.divisor_inv];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results` equals `io.z`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the multiplicative inverse of `io.y`.

        let mut is_valid = AB::Expr::zero();
        let mut expected_opcode = AB::Expr::zero();
        let mut expected_result = AB::Expr::zero();
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            is_valid += flag.into();
            expected_opcode += flag * AB::Expr::from_canonical_u32(opcode as u32);
            expected_result += flag * result;
        }
        expected_opcode -= AB::Expr::from_canonical_usize(self.offset);
        builder.assert_eq(cols.a, expected_result);
        builder.assert_bool(is_valid.clone());

        builder.assert_eq(cols.is_div, b * cols.divisor_inv);

        AdapterAirContext {
            to_pc: None,
            reads: [[cols.b.into()], [cols.c.into()]].into(),
            writes: [[cols.a.into()]].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
pub struct NewFieldArithmeticRecord<F> {
    pub opcode: FieldArithmeticOpcode,
    pub a: F,
    pub b: F,
    pub c: F,
}

#[derive(Debug)]
pub struct NewFieldArithmeticCoreChip {
    pub air: NewFieldArithmeticCoreAir,
}

impl NewFieldArithmeticCoreChip {
    pub fn new(offset: usize) -> Self {
        Self {
            air: NewFieldArithmeticCoreAir { offset },
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for NewFieldArithmeticCoreChip
where
    I::Reads: Into<[[F; 1]; 2]>,
    I::Writes: From<[[F; 1]; 1]>,
{
    type Record = NewFieldArithmeticRecord<F>;
    type Air = NewFieldArithmeticCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let local_opcode_index = FieldArithmeticOpcode::from_usize(opcode - self.air.offset);

        let data: [[F; 1]; 2] = reads.into();
        let b = data[0][0];
        let c = data[1][0];
        let a = solve_field_arithmetic(local_opcode_index, b, c).unwrap();

        let output: AdapterRuntimeContext<F, I> = AdapterRuntimeContext {
            to_pc: None,
            writes: [[a]].into(),
        };

        let record = Self::Record {
            opcode: local_opcode_index,
            a,
            b,
            c,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldArithmeticOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let NewFieldArithmeticRecord { opcode, a, b, c } = record;
        let row_slice: &mut NewFieldArithmeticCols<_> = row_slice.borrow_mut();
        row_slice.is_add = F::from_bool(opcode == FieldArithmeticOpcode::ADD);
        row_slice.is_sub = F::from_bool(opcode == FieldArithmeticOpcode::SUB);
        row_slice.is_mul = F::from_bool(opcode == FieldArithmeticOpcode::MUL);
        row_slice.is_div = F::from_bool(opcode == FieldArithmeticOpcode::DIV);
        row_slice.divisor_inv = if opcode == FieldArithmeticOpcode::DIV {
            c.inverse()
        } else {
            F::zero()
        };
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn solve_field_arithmetic<F: Field>(
    opcode: FieldArithmeticOpcode,
    b: F,
    c: F,
) -> Option<F> {
    match opcode {
        FieldArithmeticOpcode::ADD => Some(b + c),
        FieldArithmeticOpcode::SUB => Some(b - c),
        FieldArithmeticOpcode::MUL => Some(b * c),
        FieldArithmeticOpcode::DIV => {
            if c.is_zero() {
                None
            } else {
                Some(b * c.inverse())
            }
        }
    }
}
