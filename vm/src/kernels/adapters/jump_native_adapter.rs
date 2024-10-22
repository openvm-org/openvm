use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::native_adapter::{NativeReadRecord, NativeWriteRecord};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    rv32im::adapters::JumpUiProcessedInstruction,
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

/// R reads(R<=2), W writes(W<=1).
/// Operands: b for the first read, c for the second read, a for the first write.
/// If an operand is not used, its address space and pointer should be all 0.
#[derive(Clone, Debug)]
pub struct JumpNativeAdapterChip<F: Field, const R: usize, const W: usize> {
    pub air: JumpNativeAdapterAir<R, W>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const R: usize, const W: usize> JumpNativeAdapterChip<F, R, W> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: JumpNativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JumpNativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadAuxCols<T, 1>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JumpNativeAdapterWriteCols<T> {
    pub address: MemoryAddress<T, T>,
    pub write_aux: MemoryWriteAuxCols<T, 1>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JumpNativeAdapterCols<T, const R: usize, const W: usize> {
    pub from_state: ExecutionState<T>,
    pub reads_aux: [JumpNativeAdapterReadCols<T>; R],
    pub writes_aux: [JumpNativeAdapterWriteCols<T>; W],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JumpNativeAdapterAir<const R: usize, const W: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const R: usize, const W: usize> BaseAir<F> for JumpNativeAdapterAir<R, W> {
    fn width(&self) -> usize {
        JumpNativeAdapterCols::<F, R, W>::width()
    }
}

impl<AB: InteractionBuilder, const R: usize, const W: usize> VmAdapterAir<AB>
    for JumpNativeAdapterAir<R, W>
{
    type Interface =
        BasicAdapterInterface<AB::Expr, JumpUiProcessedInstruction<AB::Expr>, R, W, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &JumpNativeAdapterCols<_, R, W> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        for (i, r_cols) in cols.reads_aux.iter().enumerate() {
            self.memory_bridge
                .read(
                    r_cols.address,
                    ctx.reads[i].clone(),
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
            || MemoryAddress::new(AB::Expr::from(AB::F::zero()), AB::Expr::from(AB::F::zero()));
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
                (1, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &JumpNativeAdapterCols<_, R, W> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32, const R: usize, const W: usize> VmAdapterChip<F>
    for JumpNativeAdapterChip<F, R, W>
{
    type ReadRecord = NativeReadRecord<F, R>;
    type WriteRecord = NativeWriteRecord<F, W>;
    type Air = JumpNativeAdapterAir<R, W>;
    type Interface = BasicAdapterInterface<F, JumpUiProcessedInstruction<F>, R, W, 1, 1>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        assert!(R <= 2);
        let Instruction { b, c, e, f, .. } = *instruction;

        let mut reads = Vec::with_capacity(R);
        if R >= 1 {
            reads.push(memory.read::<1>(e, b));
        }
        if R >= 2 {
            reads.push(memory.read::<1>(f, c));
        }
        let i_reads: [_; R] = std::array::from_fn(|i| reads[i].data);

        Ok((
            i_reads,
            Self::ReadRecord {
                reads: reads.try_into().unwrap(),
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        assert!(W <= 1);
        let Instruction { a, d, .. } = *instruction;
        let mut writes = Vec::with_capacity(W);
        if W >= 1 {
            writes.push(memory.write(d, a, output.writes[0]));
        }

        Ok((
            ExecutionState {
                pc: from_state.pc + 1,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state,
                writes: writes.try_into().unwrap(),
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let row_slice: &mut JumpNativeAdapterCols<_, R, W> = row_slice.borrow_mut();

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.reads_aux = read_record.reads.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            JumpNativeAdapterReadCols {
                address,
                read_aux: aux_cols_factory.make_read_aux_cols(x),
            }
        });
        row_slice.writes_aux = write_record.writes.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            JumpNativeAdapterWriteCols {
                address,
                write_aux: aux_cols_factory.make_write_aux_cols(x),
            }
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
