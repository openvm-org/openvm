use std::{marker::PhantomData, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionOutput, IntegrationInterface,
        MachineAdapter, MachineAdapterAir, MachineAdapterInterface, Result,
    },
    memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryChip, MemoryChipRef, MemoryReadRecord, MemoryWriteRecord,
    },
    program::{bridge::ProgramBus, Instruction},
};

/// Reads instructions of the form OP a, b, c, d, e where [a:4]_d = [b:4]_d op [c:4]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[derive(Debug)]
pub struct Rv32AluAdapter<F: Field> {
    _marker: PhantomData<F>,
    pub air: Rv32AluAdapterAir,
}

impl<F: PrimeField32> Rv32AluAdapter<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            _marker: PhantomData,
            air: Rv32AluAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
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

/// Interface for reading two RV32 registers, or one RV32 register and
/// one immediate
pub struct Rv32AluAdapterInterface<T>(PhantomData<T>);

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32AluProcessedInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}

impl<T> From<(T, T)> for Rv32AluProcessedInstruction<T> {
    fn from((is_valid, opcode): (T, T)) -> Self {
        Self { is_valid, opcode }
    }
}

impl<T: AbstractField> MachineAdapterInterface<T> for Rv32AluAdapterInterface<T> {
    type Reads = [[T; RV32_REGISTER_NUM_LANES]; 2];
    type Writes = [T; RV32_REGISTER_NUM_LANES];
    type ProcessedInstruction = Rv32AluProcessedInstruction<T>;
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32AluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LANES>,
}

impl<T> Rv32AluAdapterCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32AluAdapterCols<u8>>()
    }
}

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

impl<F: PrimeField32, AB: InteractionBuilder + PairBuilder + AirBuilderWithPublicValues>
    MachineAdapterAir<F, Rv32AluAdapter<F>, AB> for Rv32AluAdapterAir
{
    fn eval_adapter_constraints(
        &self,
        builder: &mut AB,
        local: &Rv32AluAdapterCols<AB::Var>,
        interface: IntegrationInterface<AB::Expr, Rv32AluAdapterInterface<AB::Expr>>,
    ) {
        let timestamp: AB::Var = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local.rs1_ptr),
                interface.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, interface.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2_ptr),
                interface.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), local.rd_ptr),
                interface.writes,
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                &local.writes_aux,
            )
            .eval(builder, interface.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_pc(
                interface.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    AB::Expr::one(),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, interface.instruction.is_valid);
    }
}

impl<F: PrimeField32> MachineAdapter<F> for Rv32AluAdapter<F> {
    type ReadRecord = Rv32AluReadRecord<F>;
    type WriteRecord = Rv32AluWriteRecord<F>;
    type Air = Rv32AluAdapterAir;
    type Cols<T> = Rv32AluAdapterCols<T>;
    type Interface<T: AbstractField> = Rv32AluAdapterInterface<T>;

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
        let rd = memory.write(d, a, output.writes);

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
        memory: &mut MemoryChip<F>,
        row_slice: &mut Self::Cols<F>,
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        row_slice.from_state = write_record.from_state.map(F::from_canonical_usize);
        row_slice.rd_ptr = write_record.rd.pointer;
        row_slice.rs1_ptr = read_record.rs1.pointer;
        row_slice.rs2_ptr = read_record.rs2.pointer;
        // TODO: rs2_as definition
        row_slice.reads_aux = [
            aux_cols_factory.make_read_aux_cols(read_record.rs1),
            aux_cols_factory.make_read_aux_cols(read_record.rs2),
        ];
        row_slice.writes_aux = aux_cols_factory.make_write_aux_cols(write_record.rd);
    }

    fn air(&self) -> Self::Air {
        self.air
    }
}
