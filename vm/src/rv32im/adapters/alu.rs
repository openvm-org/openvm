use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::utils::not;
use ax_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::instruction::Instruction;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{
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

/// Reads instructions of the form OP a, b, c, d, e where [a:4]_d = [b:4]_d op [c:4]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[derive(Clone, Debug)]
pub struct Rv32BaseAluAdapterChip<F: Field> {
    pub air: Rv32BaseAluAdapterAir,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Rv32BaseAluAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: Rv32BaseAluAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32BaseAluReadRecord<F: Field> {
    /// Read register value from address space d=1
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>,
    /// Either
    /// - read rs2 register value or
    /// - if `rs2_is_imm` is true, this is None
    pub rs2: Option<MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>>,
    /// immediate value of rs2 or 0
    pub rs2_imm: F,
}

#[derive(Clone, Debug)]
pub struct Rv32BaseAluWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    /// Write to destination register
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LIMBS>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BaseAluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    // Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BaseAluAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
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

        // // if rs2 is an immediate value, constrain that its 4-byte representation is correct
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
                MemoryAddress::new(local.rs2_as, local.rs2),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), local.rd_ptr),
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
                    AB::Expr::one(),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (4, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BaseAluAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32BaseAluAdapterChip<F> {
    type ReadRecord = Rv32BaseAluReadRecord<F>;
    type WriteRecord = Rv32BaseAluWriteRecord<F>;
    type Air = Rv32BaseAluAdapterAir;
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
        let Instruction { b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert!(e.as_canonical_u32() <= 1);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LIMBS>(d, b);
        let (rs2, rs2_data, rs2_imm) = if e.is_zero() {
            let c_u32 = c.as_canonical_u32();
            debug_assert_eq!(c_u32 >> 24, 0);
            memory.increment_timestamp();
            (
                None,
                [
                    c_u32 as u8,
                    (c_u32 >> 8) as u8,
                    (c_u32 >> 16) as u8,
                    (c_u32 >> 16) as u8,
                ]
                .map(F::from_canonical_u8),
                c,
            )
        } else {
            let rs2_read = memory.read::<RV32_REGISTER_NUM_LIMBS>(e, c);
            (Some(rs2_read), rs2_read.data, F::zero())
        };

        Ok(([rs1.data, rs2_data], Self::ReadRecord { rs1, rs2, rs2_imm }))
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
        let row_slice: &mut Rv32BaseAluAdapterCols<_> = row_slice.borrow_mut();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.rd_ptr = write_record.rd.pointer;
        row_slice.rs1_ptr = read_record.rs1.pointer;
        row_slice.rs2 = read_record
            .rs2
            .map(|rs2| rs2.pointer)
            .unwrap_or(read_record.rs2_imm);
        row_slice.rs2_as = read_record
            .rs2
            .map(|rs2| rs2.address_space)
            .unwrap_or(F::zero());
        row_slice.reads_aux = [
            aux_cols_factory.make_read_aux_cols(read_record.rs1),
            match read_record.rs2 {
                Some(rs2_record) => aux_cols_factory.make_read_aux_cols(rs2_record),
                None => MemoryReadAuxCols::<F, RV32_REGISTER_NUM_LIMBS>::disabled(),
            },
        ];
        row_slice.writes_aux = aux_cols_factory.make_write_aux_cols(write_record.rd);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
