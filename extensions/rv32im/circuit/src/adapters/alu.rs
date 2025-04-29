use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, BasicAdapterInterface,
        ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::{
    tracing_read, tracing_read_imm, tracing_write, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use crate::adapters::{memory_read, memory_write};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BaseAluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Reads instructions of the form OP a, b, c, d, e where \[a:4\]_d = \[b:4\]_d op \[c:4\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BaseAluAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32BaseAluAdapterAir {
    fn width(&self) -> usize {
        Rv32BaseAluAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32BaseAluAdapterAir {
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
        let local: &Rv32BaseAluAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // If rs2 is an immediate value, constrain that:
        // 1. It's a 16-bit two's complement integer (stored in rs2_limbs[0] and rs2_limbs[1])
        // 2. It's properly sign-extended to 32-bits (the upper limbs must match the sign bit)
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign = rs2_limbs[2].clone();
        let rs2_imm = rs2_limbs[0].clone()
            + rs2_limbs[1].clone() * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS)
            + rs2_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV32_CELL_BITS));
        builder.assert_bool(local.rs2_as);
        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(local.rs2, rs2_imm);
        rs2_imm_when.assert_eq(rs2_sign.clone(), rs2_limbs[3].clone());
        rs2_imm_when.assert_zero(
            rs2_sign.clone()
                * (AB::Expr::from_canonical_usize((1 << RV32_CELL_BITS) - 1) - rs2_sign),
        );
        self.bitwise_lookup_bus
            .send_range(rs2_limbs[0].clone(), rs2_limbs[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // This constraint ensures that the following memory read only occurs when `is_valid == 1`.
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

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
                    local.rs2.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BaseAluAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct Rv32BaseAluAdapterStep<const LIMB_BITS: usize>;

impl<F: PrimeField32, CTX, const LIMB_BITS: usize> AdapterTraceStep<F, CTX>
    for Rv32BaseAluAdapterStep<LIMB_BITS>
{
    const WIDTH: usize = size_of::<Rv32BaseAluAdapterCols<u8>>();
    type ReadData = ([u8; RV32_REGISTER_NUM_LIMBS], [u8; RV32_REGISTER_NUM_LIMBS]);
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_REGISTER_AS || e.as_canonical_u32() == RV32_IMM_AS
        );

        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.rs1_ptr = b;
        let rs1 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            b.as_canonical_u32(),
            &mut adapter_row.reads_aux[0],
        );

        let rs2 = if e.as_canonical_u32() == RV32_REGISTER_AS {
            adapter_row.rs2_as = e;
            adapter_row.rs2 = c;

            tracing_read(
                memory,
                RV32_REGISTER_AS,
                c.as_canonical_u32(),
                &mut adapter_row.reads_aux[1],
            )
        } else {
            adapter_row.rs2_as = e;

            tracing_read_imm(memory, c.as_canonical_u32(), &mut adapter_row.rs2)
        };

        (rs1, rs2)
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.rd_ptr = a;
        tracing_write(
            memory,
            RV32_REGISTER_AS,
            a.as_canonical_u32(),
            data,
            &mut adapter_row.writes_aux,
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        bitwise_lookup_chip: &BitwiseOperationLookupChip<LIMB_BITS>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

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

impl<F, const LIMB_BITS: usize> AdapterExecutorE1<F> for Rv32BaseAluAdapterStep<LIMB_BITS>
where
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = ([u8; RV32_REGISTER_NUM_LIMBS], [u8; RV32_REGISTER_NUM_LIMBS]);
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    #[inline(always)]
    fn read<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData
    where
        Mem: GuestMemory,
    {
        let Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_IMM_AS || e.as_canonical_u32() == RV32_REGISTER_AS
        );

        let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
            memory_read(memory, RV32_REGISTER_AS, b.as_canonical_u32());

        let rs2 = if e.as_canonical_u32() == RV32_REGISTER_AS {
            let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
                memory_read(memory, RV32_REGISTER_AS, c.as_canonical_u32());
            rs2
        } else {
            let imm = c.as_canonical_u32();
            debug_assert_eq!(imm >> 24, 0);
            let mut imm_le = imm.to_le_bytes();
            imm_le[3] = imm_le[2];
            imm_le
        };

        (rs1, rs2)
    }

    #[inline(always)]
    fn write<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>, rd: &Self::WriteData)
    where
        Mem: GuestMemory,
    {
        let Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        memory_write(memory, d.as_canonical_u32(), a.as_canonical_u32(), rd);
    }
}
