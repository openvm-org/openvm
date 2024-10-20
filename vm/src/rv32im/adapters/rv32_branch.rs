use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{JumpUiProcessedInstruction, RV32_REGISTER_NUM_LANES};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

/// Reads instructions of the form OP a, b, c, d, e where if([a:4]_d op [b:4]_e) pc += c.
/// Operands d and e can only be 1.
#[derive(Debug)]
pub struct Rv32BranchAdapterChip<F: Field> {
    pub air: Rv32BranchAdapterAir,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32> Rv32BranchAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: Rv32BranchAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}

#[derive(Debug)]
pub struct Rv32BranchReadRecord<F: Field> {
    /// Read register value from address space d = 1
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    /// Read register value from address space e = 1
    pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug)]
pub struct Rv32BranchWriteRecord {
    pub from_state: ExecutionState<u32>,
}

pub type Rv32BranchAdapterInterface<T> =
    BasicAdapterInterface<T, JumpUiProcessedInstruction<T>, 2, 0, RV32_REGISTER_NUM_LANES, 0>;

/*
pub struct Rv32BranchAdapterInterface<T>(PhantomData<T>);

impl<T> VmAdapterInterface<T> for Rv32BranchAdapterInterface<T> {
    type Reads = [[T; RV32_REGISTER_NUM_LANES]; 2];
    type Writes = ();
    type ProcessedInstruction = JumpUiProcessedInstruction<T>;
}
*/

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BranchAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BranchAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32BranchAdapterAir {
    fn width(&self) -> usize {
        Rv32BranchAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32BranchAdapterAir {
    type Interface = Rv32BranchAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32BranchAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local.rs2_ptr),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::one(),
                    AB::Expr::one(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (4, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BranchAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32BranchAdapterChip<F> {
    type ReadRecord = Rv32BranchReadRecord<F>;
    type WriteRecord = Rv32BranchWriteRecord;
    type Air = Rv32BranchAdapterAir;
    type Interface = Rv32BranchAdapterInterface<F>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_a: a,
            op_b: b,
            d,
            e,
            ..
        } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 1);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LANES>(d, a);
        let rs2 = memory.read::<RV32_REGISTER_NUM_LANES>(e, b);

        Ok(([rs1.data, rs2.data], Self::ReadRecord { rs1, rs2 }))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let timestamp_delta = memory.timestamp() - from_state.timestamp;
        debug_assert!(
            timestamp_delta == 2,
            "timestamp delta is {}, expected 2",
            timestamp_delta
        );

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let row_slice: &mut Rv32BranchAdapterCols<_> = row_slice.borrow_mut();
        let aux_cols_factory = &self.aux_cols_factory;
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.rs1_ptr = read_record.rs1.pointer;
        row_slice.rs2_ptr = read_record.rs2.pointer;
        row_slice.reads_aux = [
            aux_cols_factory.make_read_aux_cols(read_record.rs1),
            aux_cols_factory.make_read_aux_cols(read_record.rs2),
        ]
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
