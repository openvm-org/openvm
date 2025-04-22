use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterRuntimeContext, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState, ImmInstruction,
        Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols},
            online::{GuestMemory, TracingMemory},
            MemoryAddress, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

use super::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Rv32BranchReadRecord {
    /// Read register value from address space d = 1
    pub rs1: RecordId,
    /// Read register value from address space e = 1
    pub rs2: RecordId,
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Rv32BranchWriteRecord {
    pub from_state: ExecutionState<u32>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BranchAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
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
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 2, 0, RV32_REGISTER_NUM_LIMBS, 0>;

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

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BranchAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

/// Reads instructions of the form OP a, b, c, d, e where if(\[a:4\]_d op \[b:4\]_e) pc += c.
/// Operands d and e can only be 1.
#[derive(derive_new::new)]
pub struct Rv32BranchAdapterStep;

impl<F: PrimeField32, CTX> AdapterTraceStep<F, CTX> for Rv32BranchAdapterStep {
    const WIDTH: usize = size_of::<Rv32BranchAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();
        let rs1_idx = b.as_canonical_u32();
        let rs1 = tracing_read_reg(
            memory,
            rs1_idx,
            (&mut adapter_row.rs1_ptr, &mut adapter_row.reads_aux[0]),
        );
        let rs2 = tracing_read_reg_or_imm(
            memory,
            e.as_canonical_u32(),
            c.as_canonical_u32(),
            &mut adapter_row.rs2_as,
            (&mut adapter_row.rs2, &mut adapter_row.reads_aux[1]),
        );
        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();
        let rd_ptr = instruction.a.as_canonical_u32();
        tracing_write_reg(
            memory,
            rd_ptr,
            data,
            (&mut adapter_row.rd_ptr, &mut adapter_row.writes_aux),
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        mem_helper: &MemoryAuxColsFactory<F>,
        bitwise_lookup_chip: &BitwiseOperationLookupChip<LIMB_BITS>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();
        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();
        mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[0].as_mut());
        timestamp += 1;
        if !adapter_row.rs2_as.is_zero() {
            mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[1].as_mut());
        } else {
            let rs2_imm = adapter_row.rs2.as_canonical_u32();
            let mask = (1 << RV32_CELL_BITS) - 1;
            bitwise_lookup_chip.request_range(rs2_imm & mask, (rs2_imm >> 8) & mask);
        }
        timestamp += 1;
        mem_helper.fill_from_prev(timestamp, adapter_row.writes_aux.as_mut());
    }
}

impl<Mem, F> AdapterExecutorE1<Mem, F> for Rv32BranchAdapterStep
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    fn read(memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction { a, b, c, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1 = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(d, a);
        let rs2 = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(e, b);

        [rs1_bytes, rs2_bytes]
    }

    fn write(memory: &mut Mem, instruction: &Instruction<F>, rd_bytes: &Self::WriteData) {
        debug_assert!(
            timestamp_delta == 2,
            "timestamp delta is {}, expected 2",
            timestamp_delta
        );
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32BranchAdapterChip<F> {
    type ReadRecord = Rv32BranchReadRecord;
    type WriteRecord = Rv32BranchWriteRecord;
    type Air = Rv32BranchAdapterAir;
    type Interface = BasicAdapterInterface<F, ImmInstruction<F>, 2, 0, RV32_REGISTER_NUM_LIMBS, 0>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1 = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(d, a);
        let rs2 = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(e, b);

        Ok((
            [
                rs1.1.map(F::from_canonical_u8),
                rs2.1.map(F::from_canonical_u8),
            ],
            Self::ReadRecord {
                rs1: rs1.0,
                rs2: rs2.0,
            },
        ))
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
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
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
        memory: &OfflineMemory<F>,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        let row_slice: &mut Rv32BranchAdapterCols<_> = row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        let rs1 = memory.record_by_id(read_record.rs1);
        let rs2 = memory.record_by_id(read_record.rs2);
        row_slice.rs1_ptr = rs1.pointer;
        row_slice.rs2_ptr = rs2.pointer;
        aux_cols_factory.generate_read_aux(rs1, &mut row_slice.reads_aux[0]);
        aux_cols_factory.generate_read_aux(rs2, &mut row_slice.reads_aux[1]);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
