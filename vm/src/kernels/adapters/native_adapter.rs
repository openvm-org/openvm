use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
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
        program::{bridge::ProgramBus, Instruction},
    },
};

pub type NativeAdapterChip<F> = GenericNativeAdapterChip<F, 2, 1>;
pub type NativeAdapterCols<T> = GenericNativeAdapterCols<T, 2, 1>;
pub type NativeAdapterAir = GenericNativeAdapterAir<2, 1>;

pub type GenericNativeAdapterInterface<T, const R: usize, const W: usize> =
    BasicAdapterInterface<T, MinimalInstruction<T>, R, W, 1, 1>;

/// R reads(R<=2), W writes(W<=1).
/// Operands: b for the first read, c for the second read, a for the first write.
/// If an operand is not used, its address space and pointer should be all 0.
#[derive(Clone, Debug)]
pub struct GenericNativeAdapterChip<F: Field, const R: usize, const W: usize> {
    pub air: GenericNativeAdapterAir<R, W>,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32, const R: usize, const W: usize> GenericNativeAdapterChip<F, R, W> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: GenericNativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
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
pub struct GenericNativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadOrImmediateAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct GenericNativeAdapterWriteCols<T> {
    pub address: MemoryAddress<T, T>,
    pub write_aux: MemoryWriteAuxCols<T, 1>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct GenericNativeAdapterCols<T, const R: usize, const W: usize> {
    pub from_state: ExecutionState<T>,
    pub reads_aux: [GenericNativeAdapterReadCols<T>; R],
    pub writes_aux: [GenericNativeAdapterWriteCols<T>; W],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct GenericNativeAdapterAir<const R: usize, const W: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const R: usize, const W: usize> BaseAir<F> for GenericNativeAdapterAir<R, W> {
    fn width(&self) -> usize {
        GenericNativeAdapterCols::<F, R, W>::width()
    }
}

impl<AB: InteractionBuilder, const R: usize, const W: usize> VmAdapterAir<AB>
    for GenericNativeAdapterAir<R, W>
{
    type Interface = GenericNativeAdapterInterface<AB::Expr, R, W>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &GenericNativeAdapterCols<_, R, W> = local.borrow();
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
                (1, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &GenericNativeAdapterCols<_, R, W> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32, const R: usize, const W: usize> VmAdapterChip<F>
    for GenericNativeAdapterChip<F, R, W>
{
    type ReadRecord = NativeReadRecord<F, R>;
    type WriteRecord = NativeWriteRecord<F, W>;
    type Air = GenericNativeAdapterAir<R, W>;
    type Interface = GenericNativeAdapterInterface<F, R, W>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        assert!(R <= 2);
        let Instruction {
            op_b: b,
            op_c: c,
            e,
            op_f: f,
            ..
        } = *instruction;

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
        let Instruction { op_a: a, d, .. } = *instruction;
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
    ) {
        let row_slice: &mut GenericNativeAdapterCols<_, R, W> = row_slice.borrow_mut();
        let aux_cols_factory = &self.aux_cols_factory;

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.reads_aux = read_record.reads.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            GenericNativeAdapterReadCols {
                address,
                read_aux: aux_cols_factory.make_read_or_immediate_aux_cols(x),
            }
        });
        row_slice.writes_aux = write_record.writes.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            GenericNativeAdapterWriteCols {
                address,
                write_aux: aux_cols_factory.make_write_aux_cols(x),
            }
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
