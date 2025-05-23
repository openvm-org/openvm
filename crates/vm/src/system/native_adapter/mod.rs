mod util;

use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use util::{tracing_read_or_imm_native, tracing_write_native, AS_NATIVE};

use super::memory::{online::TracingMemory, MemoryAuxColsFactory};
use crate::{
    arch::{execution_mode::E1E2ExecutionCtx, AdapterExecutorE1, AdapterTraceStep, VmStateMut},
    system::memory::online::GuestMemory,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadOrImmediateAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterWriteCols<T> {
    pub address: MemoryAddress<T, T>,
    pub write_aux: MemoryWriteAuxCols<T, 1>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterCols<T, const R: usize, const W: usize> {
    pub from_state: ExecutionState<T>,
    pub reads_aux: [NativeAdapterReadCols<T>; R],
    pub writes_aux: [NativeAdapterWriteCols<T>; W],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeAdapterAir<const R: usize, const W: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const R: usize, const W: usize> BaseAir<F> for NativeAdapterAir<R, W> {
    fn width(&self) -> usize {
        NativeAdapterCols::<F, R, W>::width()
    }
}

impl<AB: InteractionBuilder, const R: usize, const W: usize> VmAdapterAir<AB>
    for NativeAdapterAir<R, W>
{
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, R, W, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeAdapterCols<_, R, W> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        for (i, r_cols) in cols.reads_aux.iter().enumerate() {
            self.memory_bridge
                .read_or_immediate(
                    r_cols.address,
                    ctx.reads[i][0].clone(),
                    timestamp_pp(),
                    &r_cols.read_aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }
        for (i, w_cols) in cols.writes_aux.iter().enumerate() {
            self.memory_bridge
                .write(
                    w_cols.address,
                    ctx.writes[i].clone(),
                    timestamp_pp(),
                    &w_cols.write_aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let zero_address =
            || MemoryAddress::new(AB::Expr::from(AB::F::ZERO), AB::Expr::from(AB::F::ZERO));
        let f = |var_addr: MemoryAddress<AB::Var, AB::Var>| -> MemoryAddress<AB::Expr, AB::Expr> {
            MemoryAddress::new(var_addr.address_space.into(), var_addr.pointer.into())
        };

        let addr_a = if W >= 1 {
            f(cols.writes_aux[0].address)
        } else {
            zero_address()
        };
        let addr_b = if R >= 1 {
            f(cols.reads_aux[0].address)
        } else {
            zero_address()
        };
        let addr_c = if R >= 2 {
            f(cols.reads_aux[1].address)
        } else {
            zero_address()
        };
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    addr_a.pointer,
                    addr_b.pointer,
                    addr_c.pointer,
                    addr_a.address_space,
                    addr_b.address_space,
                    addr_c.address_space,
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeAdapterCols<_, R, W> = local.borrow();
        cols.from_state.pc
    }
}

/// R reads(R<=2), W writes(W<=1).
/// Operands: b for the first read, c for the second read, a for the first write.
/// If an operand is not used, its address space and pointer should be all 0.
#[derive(Debug, derive_new::new)]
pub struct NativeAdapterStep<F, const R: usize, const W: usize> {
    _phantom: PhantomData<F>,
}

impl<F, CTX, const R: usize, const W: usize> AdapterTraceStep<F, CTX> for NativeAdapterStep<F, R, W>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<NativeAdapterCols<u8, R, W>>();
    type ReadData = [[F; 1]; R];
    type WriteData = [[F; 1]; W];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut NativeAdapterCols<F, R, W> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        assert!(R <= 2);

        let &Instruction { b, e, f, c, .. } = instruction;

        let cols: &mut NativeAdapterCols<_, R, W> = adapter_row.borrow_mut();

        let mut reads = [[F::ZERO; 1]; R];
        if R >= 1 {
            cols.reads_aux[0].address.pointer = b;
            reads[0][0] = tracing_read_or_imm_native(
                memory,
                e.as_canonical_u32(),
                b,
                &mut cols.reads_aux[0].address.address_space,
                &mut cols.reads_aux[0].read_aux,
            );
        }
        if R >= 2 {
            cols.reads_aux[1].address.pointer = c;
            reads[1][0] = tracing_read_or_imm_native(
                memory,
                f.as_canonical_u32(),
                c,
                &mut cols.reads_aux[1].address.address_space,
                &mut cols.reads_aux[1].read_aux,
            );
        }
        reads
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        assert!(W <= 1);

        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS_NATIVE);

        let cols: &mut NativeAdapterCols<_, R, W> = adapter_row.borrow_mut();

        if W >= 1 {
            cols.writes_aux[0].address.address_space = F::from_canonical_u32(AS_NATIVE);
            cols.writes_aux[0].address.pointer = a;
            tracing_write_native(
                memory,
                a.as_canonical_u32(),
                &data[0],
                &mut cols.writes_aux[0].write_aux,
            );
        }
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut NativeAdapterCols<_, R, W> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        for read_aux in &mut adapter_row.reads_aux {
            mem_helper.fill_from_prev(timestamp, &mut read_aux.read_aux.base);
            timestamp += 1;

            if read_aux.address.address_space.is_zero() {
                read_aux.read_aux.is_immediate = F::ONE;
                read_aux.read_aux.is_zero_aux = F::ZERO;
            } else {
                read_aux.read_aux.is_immediate = F::ZERO;
                read_aux.read_aux.is_zero_aux = read_aux.address.address_space.inverse();
            }
        }

        for write_aux in &mut adapter_row.writes_aux {
            mem_helper.fill_from_prev(timestamp, write_aux.write_aux.as_mut());
            timestamp += 1;
        }
    }
}

impl<F, const R: usize, const W: usize> AdapterExecutorE1<F> for NativeAdapterStep<F, R, W>
where
    F: PrimeField32,
{
    type ReadData = [F; R];
    type WriteData = [F; W];

    #[inline(always)]
    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        assert!(R <= 2);

        let &Instruction { b, c, e, f, .. } = instruction;

        let mut reads = [F::ZERO; R];
        if R >= 1 {
            let [value] = unsafe {
                state
                    .memory
                    .read::<F, 1>(e.as_canonical_u32(), b.as_canonical_u32())
            };
            reads[0] = value;
        }
        if R >= 2 {
            let [value] = unsafe {
                state
                    .memory
                    .read::<F, 1>(f.as_canonical_u32(), c.as_canonical_u32())
            };
            reads[1] = value;
        }
        reads
    }

    #[inline(always)]
    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
        assert!(W <= 1);

        let &Instruction { a, d, .. } = instruction;
        if W >= 1 {
            unsafe {
                state
                    .memory
                    .write(d.as_canonical_u32(), a.as_canonical_u32(), data)
            };
        }
    }
}
