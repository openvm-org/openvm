use std::{
    array,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterRuntimeContext, InsExecutorE1,
        MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip, VmExecutionState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::FieldExtensionOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

pub const BETA: usize = 11;
pub const EXT_DEG: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FieldExtensionCoreCols<T> {
    pub x: [T; EXT_DEG],
    pub y: [T; EXT_DEG],
    pub z: [T; EXT_DEG],

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: [T; EXT_DEG],
}

#[derive(Copy, Clone, Debug)]
pub struct FieldExtensionCoreAir {}

impl<F: Field> BaseAir<F> for FieldExtensionCoreAir {
    fn width(&self) -> usize {
        FieldExtensionCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldExtensionCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FieldExtensionCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; EXT_DEG]; 2]>,
    I::Writes: From<[[AB::Expr; EXT_DEG]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FieldExtensionCoreCols<_> = local_core.borrow();

        let flags = [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div];
        let opcodes = [FE4ADD, FE4SUB, BBE4MUL, BBE4DIV];
        let results = [
            FieldExtension::add(cols.y, cols.z),
            FieldExtension::subtract(cols.y, cols.z),
            FieldExtension::multiply(cols.y, cols.z),
            FieldExtension::multiply(cols.y, cols.divisor_inv),
        ];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results[:,j]` equals `io.z[j]` for each `j`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the inverse of `io.y`.

        let mut is_valid = AB::Expr::ZERO;
        let mut expected_opcode = AB::Expr::ZERO;
        let mut expected_result = [
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            is_valid += flag.into();
            expected_opcode += flag * AB::F::from_canonical_usize(opcode.local_usize());

            for (j, result_part) in result.into_iter().enumerate() {
                expected_result[j] += flag * result_part;
            }
        }

        for (x_j, expected_result_j) in izip!(cols.x, expected_result) {
            builder.assert_eq(x_j, expected_result_j);
        }
        builder.assert_bool(is_valid.clone());

        // constrain aux.divisor_inv: z * z^(-1) = 1
        let z_times_z_inv = FieldExtension::multiply(cols.z, cols.divisor_inv);
        for (i, prod_i) in z_times_z_inv.into_iter().enumerate() {
            if i == 0 {
                builder.assert_eq(cols.is_div, prod_i);
            } else {
                builder.assert_zero(prod_i);
            }
        }

        AdapterAirContext {
            to_pc: None,
            reads: [cols.y.map(Into::into), cols.z.map(Into::into)].into(),
            writes: [cols.x.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        FieldExtensionOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct FieldExtensionRecord<F> {
    pub opcode: FieldExtensionOpcode,
    pub x: [F; EXT_DEG],
    pub y: [F; EXT_DEG],
    pub z: [F; EXT_DEG],
}

pub struct FieldExtensionStep<A> {
    phantom: PhantomData<A>,
}

impl Default for FieldExtensionStep {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtensionChip {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn execute_trace_core<F>(
        &self,
        instruction: &Instruction<F>,
        [x, y]: [[u8; NUM_LIMBS]; 2],
        core_row: &mut [F],
    ) -> [u8; NUM_LIMBS]
    where
        F: PrimeField32,
    {
        todo!("Implement execute_trace_core")
    }

    pub fn fill_trace_row_core<F>(&self, core_row: &mut [F])
    where
        F: PrimeField32,
    {
        todo!("Implement fill_trace_row_core")
    }
}

impl<F, CTX, A> TraceStep<F, CTX> for FieldExtensionStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = [[u8; NUM_LIMBS]; 2],
            WriteData = [u8; NUM_LIMBS],
            TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>,
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldExtensionOpcode::from_usize(opcode - FieldExtensionOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        row_slice: &mut [F],
    ) -> Result<()> {
        todo!("Implement execute")
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        todo!("Implement fill_trace_row")
    }
}

impl<Mem, Ctx, F, A> StepExecutorE1<Mem, Ctx, F> for FieldExtensionStep<A>
where
    Mem: GuestMemory,
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<
            Mem,
            F,
            ReadData = ([F; EXT_DEG], [F; EXT_DEG]),
            WriteData = [F; EXT_DEG],
        >,
{
    fn execute_e1(
        &mut self,
        state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        let local_opcode_idx = opcode.local_opcode_idx(FieldExtensionOpcode::CLASS_OFFSET);

        let (y_val, z_val) = A::read(&mut state.memory, instruction);

        let x_val = FieldExtension::solve(
            FieldExtensionOpcode::from_usize(local_opcode_idx),
            y_val,
            z_val,
        )
        .unwrap();

        A::write(&mut state.memory, instruction, &x_val);

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for FieldExtensionCoreChip
// where
//     I::Reads: Into<[[F; EXT_DEG]; 2]>,
//     I::Writes: From<[[F; EXT_DEG]; 1]>,
// {
//     type Record = FieldExtensionRecord<F>;
//     type Air = FieldExtensionCoreAir;

//     #[allow(clippy::type_complexity)]
//     fn execute_instruction(
//         &self,
//         instruction: &Instruction<F>,
//         _from_pc: u32,
//         reads: I::Reads,
//     ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
//         let Instruction { opcode, .. } = instruction;
//         let local_opcode_idx = opcode.local_opcode_idx(FieldExtensionOpcode::CLASS_OFFSET);

//         let data: [[F; EXT_DEG]; 2] = reads.into();
//         let y: [F; EXT_DEG] = data[0];
//         let z: [F; EXT_DEG] = data[1];

//         let x = FieldExtension::solve(FieldExtensionOpcode::from_usize(local_opcode_idx), y, z)
//             .unwrap();

//         let output = AdapterRuntimeContext {
//             to_pc: None,
//             writes: [x].into(),
//         };

//         let record = Self::Record {
//             opcode: FieldExtensionOpcode::from_usize(local_opcode_idx),
//             x,
//             y,
//             z,
//         };

//         Ok((output, record))
//     }

//     fn get_opcode_name(&self, opcode: usize) -> String {
//         format!(
//             "{:?}",
//             FieldExtensionOpcode::from_usize(opcode - FieldExtensionOpcode::CLASS_OFFSET)
//         )
//     }

//     fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
//         let FieldExtensionRecord { opcode, x, y, z } = record;
//         let cols: &mut FieldExtensionCoreCols<_> = row_slice.borrow_mut();
//         cols.x = x;
//         cols.y = y;
//         cols.z = z;
//         cols.is_add = F::from_bool(opcode == FieldExtensionOpcode::FE4ADD);
//         cols.is_sub = F::from_bool(opcode == FieldExtensionOpcode::FE4SUB);
//         cols.is_mul = F::from_bool(opcode == FieldExtensionOpcode::BBE4MUL);
//         cols.is_div = F::from_bool(opcode == FieldExtensionOpcode::BBE4DIV);
//         cols.divisor_inv = if opcode == FieldExtensionOpcode::BBE4DIV {
//             FieldExtension::invert(z)
//         } else {
//             [F::ZERO; EXT_DEG]
//         };
//     }

//     fn air(&self) -> &Self::Air {
//         &self.air
//     }
// }

pub struct FieldExtension;
impl FieldExtension {
    pub(super) fn solve<F: Field>(
        opcode: FieldExtensionOpcode,
        x: [F; EXT_DEG],
        y: [F; EXT_DEG],
    ) -> Option<[F; EXT_DEG]> {
        match opcode {
            FieldExtensionOpcode::FE4ADD => Some(Self::add(x, y)),
            FieldExtensionOpcode::FE4SUB => Some(Self::subtract(x, y)),
            FieldExtensionOpcode::BBE4MUL => Some(Self::multiply(x, y)),
            FieldExtensionOpcode::BBE4DIV => Some(Self::divide(x, y)),
        }
    }

    pub(crate) fn add<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        V: Copy,
        V: Add<V, Output = E>,
    {
        array::from_fn(|i| x[i] + y[i])
    }

    pub(crate) fn subtract<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        V: Copy,
        V: Sub<V, Output = E>,
    {
        array::from_fn(|i| x[i] - y[i])
    }

    pub(crate) fn multiply<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        E: FieldAlgebra,
        V: Copy,
        V: Mul<V, Output = E>,
        E: Mul<V, Output = E>,
        V: Add<V, Output = E>,
        E: Add<V, Output = E>,
    {
        let [x0, x1, x2, x3] = x;
        let [y0, y1, y2, y3] = y;
        [
            x0 * y0 + (x1 * y3 + x2 * y2 + x3 * y1) * E::from_canonical_usize(BETA),
            x0 * y1 + x1 * y0 + (x2 * y3 + x3 * y2) * E::from_canonical_usize(BETA),
            x0 * y2 + x1 * y1 + x2 * y0 + (x3 * y3) * E::from_canonical_usize(BETA),
            x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
        ]
    }

    pub(crate) fn divide<F: Field>(x: [F; EXT_DEG], y: [F; EXT_DEG]) -> [F; EXT_DEG] {
        Self::multiply(x, Self::invert(y))
    }

    pub(crate) fn invert<F: Field>(a: [F; EXT_DEG]) -> [F; EXT_DEG] {
        // Let a = (a0, a1, a2, a3) represent the element we want to invert.
        // Define a' = (a0, -a1, a2, -a3).  By construction, the product b = a * a' will have zero
        // degree-1 and degree-3 coefficients.
        // Let b = (b0, 0, b2, 0) and define b' = (b0, 0, -b2, 0).
        // Note that c = b * b' = b0^2 - BETA * b2^2, which is an element of the base field.
        // Therefore, the inverse of a is 1 / a = a' / (a * a') = a' * b' / (b * b') = a' * b' / c.

        let [a0, a1, a2, a3] = a;

        let beta = F::from_canonical_usize(BETA);

        let mut b0 = a0 * a0 - beta * (F::TWO * a1 * a3 - a2 * a2);
        let mut b2 = F::TWO * a0 * a2 - a1 * a1 - beta * a3 * a3;

        let c = b0 * b0 - beta * b2 * b2;
        let inv_c = c.inverse();

        b0 *= inv_c;
        b2 *= inv_c;

        [
            a0 * b0 - a2 * b2 * beta,
            -a1 * b0 + a3 * b2 * beta,
            -a0 * b2 + a2 * b0,
            a1 * b2 - a3 * b0,
        ]
    }
}
