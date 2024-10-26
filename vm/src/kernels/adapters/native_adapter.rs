use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};

/// R reads(R<=2), W writes(W<=1).
/// Operands: b for the first read, c for the second read, a for the first write.
/// If an operand is not used, its address space and pointer should be all 0.
#[derive(Clone, Debug)]
pub struct NativeAdapterChip<F: Field, const R: usize, const W: usize> {
    pub air: NativeAdapterAir<R, W>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const R: usize, const W: usize> NativeAdapterChip<F, R, W> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: NativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct NativeReadRecord<F: Field, const R: usize> {
    pub reads: [MemoryReadRecord<F, 1>; R],
}

impl<F: Field, const R: usize> NativeReadRecord<F, R> {
    pub fn b(&self) -> &MemoryReadRecord<F, 1> {
        &self.reads[0]
    }

    pub fn c(&self) -> &MemoryReadRecord<F, 1> {
        &self.reads[1]
    }
}

#[derive(Debug)]
pub struct NativeWriteRecord<F: Field, const W: usize> {
    pub from_state: ExecutionState<u32>,
    pub writes: [MemoryWriteRecord<F, 1>; W],
}

impl<F: Field, const W: usize> NativeWriteRecord<F, W> {
    pub fn a(&self) -> &MemoryWriteRecord<F, 1> {
        &self.writes[0]
    }
}

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
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeAdapterCols<_, R, W> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32, const R: usize, const W: usize> VmAdapterChip<F>
    for NativeAdapterChip<F, R, W>
{
    type ReadRecord = NativeReadRecord<F, R>;
    type WriteRecord = NativeWriteRecord<F, W>;
    type Air = NativeAdapterAir<R, W>;
    type Interface = BasicAdapterInterface<F, MinimalInstruction<F>, R, W, 1, 1>;

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
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
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
        let row_slice: &mut NativeAdapterCols<_, R, W> = row_slice.borrow_mut();

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.reads_aux = read_record.reads.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            NativeAdapterReadCols {
                address,
                read_aux: aux_cols_factory.make_read_or_immediate_aux_cols(x),
            }
        });
        row_slice.writes_aux = write_record.writes.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            NativeAdapterWriteCols {
                address,
                write_aux: aux_cols_factory.make_write_aux_cols(x),
            }
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
