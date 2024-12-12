use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{AbstractField, Field, PrimeField32},
};

use super::RV32_REGISTER_NUM_LIMBS;

/// Reads instructions of the form OP a, b, c, d where [a:4]_d = [b:4]_d op [c:4]_d.
/// Operand d can only be 1, and there is no immediate support.
#[derive(Debug)]
pub struct Rv32MultAdapterChip<F: Field> {
    pub air: Rv32MultAdapterAir,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Rv32MultAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        Self {
            air: Rv32MultAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge: memory_controller.memory_bridge(),
            },
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct Rv32MultReadRecord<F: Field> {
    /// Reads from operand registers
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>,
    pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Debug)]
pub struct Rv32MultWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    /// Write to destination register
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LIMBS>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32MultAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32MultAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32MultAdapterAir {
    fn width(&self) -> usize {
        Rv32MultAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32MultAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32MultAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs2_ptr),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rd_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (4, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32MultAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32MultAdapterChip<F> {
    type ReadRecord = Rv32MultReadRecord<F>;
    type WriteRecord = Rv32MultWriteRecord<F>;
    type Air = Rv32MultAdapterAir;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        2,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, d, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LIMBS>(d, b);
        let rs2 = memory.read::<RV32_REGISTER_NUM_LIMBS>(d, c);

        Ok(([rs1.data, rs2.data], Self::ReadRecord { rs1, rs2 }))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction { a, d, .. } = *instruction;
        let rd = memory.write(d, a, output.writes[0]);

        let timestamp_delta = memory.timestamp() - from_state.timestamp;
        debug_assert!(
            timestamp_delta == 3,
            "timestamp delta is {}, expected 3",
            timestamp_delta
        );

        Ok((
            ExecutionState {
                pc: from_state.pc + 4,
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
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let row_slice: &mut Rv32MultAdapterCols<_> = row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.rd_ptr = write_record.rd.pointer;
        row_slice.rs1_ptr = read_record.rs1.pointer;
        row_slice.rs2_ptr = read_record.rs2.pointer;
        row_slice.reads_aux = [
            aux_cols_factory.make_read_aux_cols(read_record.rs1),
            aux_cols_factory.make_read_aux_cols(read_record.rs2),
        ];
        row_slice.writes_aux = aux_cols_factory.make_write_aux_cols(write_record.rd);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
