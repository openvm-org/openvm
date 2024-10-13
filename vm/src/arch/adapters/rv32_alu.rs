use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    mem::size_of,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{Rv32RTypeAdapterInterface, RV32_REGISTER_NUM_LANES};
use crate::{
    arch::{
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionOutput, IntegrationInterface,
        MachineAdapter, MachineAdapterAir, MachineAdapterInterface, Result,
    },
    memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory, MemoryChip, MemoryChipRef, MemoryReadRecord,
        MemoryWriteRecord,
    },
    program::{bridge::ProgramBus, Instruction},
};

/// Reads instructions of the form OP a, b, c, d, e where [a:4]_d = [b:4]_d op [c:4]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Rv32AluAdapter<F: Field> {
    pub air: Rv32AluAdapterAir,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32> Rv32AluAdapter<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        let memory_chip = RefCell::borrow(&memory_chip);
        let memory_bridge = memory_chip.memory_bridge();
        let aux_cols_factory = memory_chip.aux_cols_factory();
        Self {
            air: Rv32AluAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}

#[derive(Debug)]
pub struct Rv32AluReadRecord<F: Field> {
    /// Read register value from address space d=1
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    /// Either
    /// - read rs2 register value or
    /// - if `rs2_is_imm` is true, then this is a dummy read where `data` is used for handling of immediate.
    pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    pub rs2_is_imm: bool,
}

#[derive(Debug)]
pub struct Rv32AluWriteRecord<F: Field> {
    pub from_state: ExecutionState<usize>,
    /// Write to destination register
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32AluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_idx: T,
    pub rs1_idx: T,
    pub rs2_idx: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LANES>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32AluAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32AluAdapterAir {
    fn width(&self) -> usize {
        size_of::<Rv32AluAdapterCols<u8>>()
    }
}

impl<AB: InteractionBuilder> MachineAdapterAir<AB> for Rv32AluAdapterAir {
    type Interface = Rv32RTypeAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: IntegrationInterface<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32AluAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local.rs1_idx),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2_idx),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), local.rd_idx),
                ctx.writes[0].clone(),
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_pc(
                ctx.instruction.opcode,
                [
                    local.rd_idx.into(),
                    local.rs1_idx.into(),
                    local.rs2_idx.into(),
                    AB::Expr::one(),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, ctx.instruction.is_valid);
    }
}

impl<F: PrimeField32> MachineAdapter<F> for Rv32AluAdapter<F> {
    type ReadRecord = Rv32AluReadRecord<F>;
    type WriteRecord = Rv32AluWriteRecord<F>;
    type Air = Rv32AluAdapterAir;
    type Interface<T: AbstractField> = Rv32RTypeAdapterInterface<T>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as MachineAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_b: b,
            op_c: c,
            d,
            e,
            ..
        } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert!(e.as_canonical_u32() <= 1);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LANES>(d, b);
        let rs2_is_imm = e.is_zero();
        let rs2 = if rs2_is_imm {
            let c_u32 = (c).as_canonical_u32();
            debug_assert_eq!(c_u32 >> 24, 0);
            let c_bytes_le = [
                c_u32 as u8,
                (c_u32 >> 8) as u8,
                (c_u32 >> 16) as u8,
                (c_u32 >> 16) as u8,
            ];
            MemoryReadRecord {
                address_space: F::zero(),
                pointer: F::zero(),
                timestamp: F::zero(),
                prev_timestamp: F::zero(),
                data: c_bytes_le.map(F::from_canonical_u8),
            }
        } else {
            memory.read::<RV32_REGISTER_NUM_LANES>(e, c)
        };

        Ok((
            [rs1.data, rs2.data],
            Self::ReadRecord {
                rs1,
                rs2,
                rs2_is_imm,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: InstructionOutput<F, Self::Interface<F>>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        // TODO: timestamp delta debug check

        let Instruction { op_a: a, d, .. } = *instruction;
        let rd = memory.write(d, a, output.writes[0]);

        Ok((
            ExecutionState {
                pc: from_state.pc + 4,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
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
        let row_slice: &mut Rv32AluAdapterCols<_> = row_slice.borrow_mut();
        let aux_cols_factory = &self.aux_cols_factory;
        row_slice.from_state = write_record.from_state.map(F::from_canonical_usize);
        row_slice.rd_idx = write_record.rd.pointer;
        row_slice.rs1_idx = read_record.rs1.pointer;
        row_slice.rs2_idx = read_record.rs2.pointer;
        // TODO: rs2_as definition
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
