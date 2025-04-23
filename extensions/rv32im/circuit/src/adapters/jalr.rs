use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterRuntimeContext, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState, Result,
        SignedImmInstruction, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            online::{GuestMemory, TracingMemory},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

use super::{tmp_convert_to_u8s, RV32_REGISTER_NUM_LIMBS};

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32JalrReadRecord {
    pub rs1: RecordId,
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32JalrWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalrAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// Only writes if `needs_write`.
    /// Sets `needs_write` to 0 iff `rd == x0`
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
    type Interface = BasicAdapterInterface<
        AB::Expr,
        SignedImmInstruction<AB::Expr>,
        1,
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
        let local_cols: &Rv32JalrAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let write_count = local_cols.needs_write;

        builder.assert_bool(write_count);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs1_ptr,
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd_ptr,
                ),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));

        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                    write_count.into(),
                    ctx.instruction.imm_sign,
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

// This adapter reads from [b:4]_d (rs1) and writes to [a:4]_d (rd)
#[derive(derive_new::new)]
pub struct Rv32JalrAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32JalrAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32JalrAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        todo!("Implement read method");
    }

    #[inline(always)]
    fn write(
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        todo!("Implement write method");
    }

    #[inline(always)]
    fn fill_trace_row(
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        todo!("Implement fill_trace_row method");
    }
}

impl<Mem, F> AdapterExecutorE1<Mem, F> for Rv32JalrAdapterStep
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = [u8; RV32_REGISTER_NUM_LIMBS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    fn read(memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction { b, d, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { memory.read(d.as_canonical_u32(), b.as_canonical_u32()) };

        rs1
    }

    fn write(memory: &mut Mem, instruction: &Instruction<F>, data: &Self::WriteData) {
        let Instruction {
            a, d, f: enabled, ..
        } = instruction;

        if *enabled != F::ZERO {
            unsafe {
                memory.write(d.as_canonical_u32(), a.as_canonical_u32(), data);
            }
        }
    }
}

// impl<F: PrimeField32> VmAdapterChip<F> for Rv32JalrAdapterChip<F> {
//     type ReadRecord = Rv32JalrReadRecord;
//     type WriteRecord = Rv32JalrWriteRecord;
//     type Air = Rv32JalrAdapterAir;
//     type Interface = BasicAdapterInterface<
//         F,
//         SignedImmInstruction<F>,
//         1,
//         1,
//         RV32_REGISTER_NUM_LIMBS,
//         RV32_REGISTER_NUM_LIMBS,
//     >;
//     fn preprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//     ) -> Result<(
//         <Self::Interface as VmAdapterInterface<F>>::Reads,
//         Self::ReadRecord,
//     )> {
//         let Instruction { b, d, .. } = *instruction;
//         debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

//         let rs1 = memory.read::<u8, RV32_REGISTER_NUM_LIMBS>(d, b);

//         Ok((
//             [rs1.1.map(F::from_canonical_u8)],
//             Rv32JalrReadRecord { rs1: rs1.0 },
//         ))
//     }

//     fn postprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//         from_state: ExecutionState<u32>,
//         output: AdapterRuntimeContext<F, Self::Interface>,
//         _read_record: &Self::ReadRecord,
//     ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
//         let Instruction {
//             a, d, f: enabled, ..
//         } = *instruction;
//         let rd_id = if enabled != F::ZERO {
//             let (record_id, _) = memory.write(d, a, &tmp_convert_to_u8s(output.writes[0]));
//             Some(record_id)
//         } else {
//             memory.increment_timestamp();
//             None
//         };

//         Ok((
//             ExecutionState {
//                 pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
//                 timestamp: memory.timestamp(),
//             },
//             Self::WriteRecord { from_state, rd_id },
//         ))
//     }

//     fn generate_trace_row(
//         &self,
//         row_slice: &mut [F],
//         read_record: Self::ReadRecord,
//         write_record: Self::WriteRecord,
//         memory: &OfflineMemory<F>,
//     ) {
//         let aux_cols_factory = memory.aux_cols_factory();
//         let adapter_cols: &mut Rv32JalrAdapterCols<_> = row_slice.borrow_mut();
//         adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
//         let rs1 = memory.record_by_id(read_record.rs1);
//         adapter_cols.rs1_ptr = rs1.pointer;
//         aux_cols_factory.generate_read_aux(rs1, &mut adapter_cols.rs1_aux_cols);
//         if let Some(id) = write_record.rd_id {
//             let rd = memory.record_by_id(id);
//             adapter_cols.rd_ptr = rd.pointer;
//             adapter_cols.needs_write = F::ONE;
//             aux_cols_factory.generate_write_aux(rd, &mut adapter_cols.rd_aux_cols);
//         }
//     }

//     fn air(&self) -> &Self::Air {
//         &self.air
//     }
// }
