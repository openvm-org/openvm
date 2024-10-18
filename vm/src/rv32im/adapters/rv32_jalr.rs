use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
};

use afs_derive::AlignedBorrow;
use afs_primitives::utils::not;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::{JumpUiProcessedInstruction, RV32_REGISTER_NUM_LANES};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

type Rv32JalrAdapterInterface<T> = BasicAdapterInterface<
    T,
    JumpUiProcessedInstruction<T>,
    1,
    1,
    RV32_REGISTER_NUM_LANES,
    RV32_REGISTER_NUM_LANES,
>;

// This adapter reads from [b:4]_d (rs1) and writes to [a:4]_d (rd)
#[derive(Debug, Clone)]
pub struct Rv32JalrAdapterChip<F: Field> {
    pub air: Rv32JalrAdapterAir,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32> Rv32JalrAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: Rv32JalrAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}
#[derive(Debug, Clone)]
pub struct Rv32JalrReadRecord<F: Field> {
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug, Clone)]
pub struct Rv32JalrWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub rd: Option<MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalrAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32JalrAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for Rv32JalrAdapterAir {
    fn width(&self) -> usize {
        Rv32JalrAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32JalrAdapterAir {
    type Interface = Rv32JalrAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32JalrAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let (write_count, op_f) = (local_cols.needs_write, local_cols.needs_write);

        builder.assert_bool(write_count);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count);

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local_cols.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), local_cols.rd_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(4));

        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::one(),
                    AB::Expr::zero(),
                    op_f.into(),
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32JalrAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32JalrAdapterChip<F> {
    type ReadRecord = Rv32JalrReadRecord<F>;
    type WriteRecord = Rv32JalrWriteRecord<F>;
    type Air = Rv32JalrAdapterAir;
    type Interface = Rv32JalrAdapterInterface<F>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { op_b: b, d, .. } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), 1);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LANES>(d, b);

        Ok(([rs1.data], Rv32JalrReadRecord { rs1 }))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction {
            op_a: a,
            d,
            op_f: enabled,
            ..
        } = *instruction;
        let rd = if enabled != F::zero() {
            Some(memory.write(d, a, output.writes[0]))
        } else {
            memory.increment_timestamp();
            None
        };

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state, rd },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let adapter_cols: &mut Rv32JalrAdapterCols<_> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.rs1_ptr = read_record.rs1.pointer;
        adapter_cols.rs1_aux_cols = self.aux_cols_factory.make_read_aux_cols(read_record.rs1);
        (adapter_cols.rd_ptr, adapter_cols.rd_aux_cols) = if let Some(rd) = write_record.rd {
            adapter_cols.needs_write = F::one();
            (rd.pointer, self.aux_cols_factory.make_write_aux_cols(rd))
        } else {
            adapter_cols.needs_write = F::zero();
            (F::zero(), MemoryWriteAuxCols::disabled())
        };
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
