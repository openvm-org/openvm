use std::{
    array,
    borrow::{Borrow, BorrowMut},
    ops::{Add, Mul, Sub},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, MinimalInstruction, Result,
        StepExecutorE1, TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
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

#[derive(derive_new::new, Copy, Clone, Debug)]
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

#[derive(derive_new::new)]
pub struct FieldExtensionCoreStep<A> {
    adapter: A,
}

impl<F, CTX, A> TraceStep<F, CTX> for FieldExtensionCoreStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = [[F; EXT_DEG]; 2],
            WriteData = [F; EXT_DEG],
            TraceContext<'a> = (),
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
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let &Instruction { opcode, .. } = instruction;

        let local_opcode = FieldExtensionOpcode::from_usize(
            opcode.local_opcode_idx(FieldExtensionOpcode::CLASS_OFFSET),
        );

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let [y, z] = self.adapter.read(state.memory, instruction, adapter_row);

        let x = FieldExtension::solve(local_opcode, y, z).unwrap();

        let core_row: &mut FieldExtensionCoreCols<_> = core_row.borrow_mut();
        core_row.x = x;
        core_row.y = y;
        core_row.z = z;
        core_row.is_add = F::from_bool(local_opcode == FieldExtensionOpcode::FE4ADD);
        core_row.is_sub = F::from_bool(local_opcode == FieldExtensionOpcode::FE4SUB);
        core_row.is_mul = F::from_bool(local_opcode == FieldExtensionOpcode::BBE4MUL);
        core_row.is_div = F::from_bool(local_opcode == FieldExtensionOpcode::BBE4DIV);

        self.adapter
            .write(state.memory, instruction, adapter_row, &x);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);

        let core_row: &mut FieldExtensionCoreCols<_> = core_row.borrow_mut();

        core_row.divisor_inv = if core_row.is_div.is_one() {
            FieldExtension::invert(core_row.z)
        } else {
            [F::ZERO; EXT_DEG]
        };
    }
}

impl<F, A> StepExecutorE1<F> for FieldExtensionCoreStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<F, ReadData = [[F; EXT_DEG]; 2], WriteData = [F; EXT_DEG]>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { opcode, .. } = instruction;

        let local_opcode_idx = opcode.local_opcode_idx(FieldExtensionOpcode::CLASS_OFFSET);

        let [y_val, z_val] = self.adapter.read(state, instruction);

        let x_val = FieldExtension::solve(
            FieldExtensionOpcode::from_usize(local_opcode_idx),
            y_val,
            z_val,
        )
        .unwrap();

        self.adapter.write(state, instruction, &x_val);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

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
