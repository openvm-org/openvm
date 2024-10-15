use std::{marker::PhantomData, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryController, MemoryControllerRef, MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

/// Reads instructions of the form OP a, b, c, d where [a:4]_d = [b:4]_d op [c:4]_d.
/// Operand d can only be 1, and there is no immediate support.
#[derive(Debug)]
pub struct Rv32MultAdapter<F: Field> {
    _marker: PhantomData<F>,
    pub air: Rv32MultAdapterAir,
}

impl<F: PrimeField32> Rv32MultAdapter<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_bridge = memory_controller.borrow().memory_bridge();
        Self {
            _marker: PhantomData,
            air: Rv32MultAdapterAir {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _memory_bridge: memory_bridge,
            },
        }
    }
}

#[derive(Debug)]
pub struct Rv32MultReadRecord<F: Field> {
    /// Reads from operand registers
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug)]
pub struct Rv32MultWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    /// Write to destination register
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>,
}

/// Interface for reading two RV32 registers
pub struct Rv32MultAdapterInterface<T>(PhantomData<T>);

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32MultProcessedInstruction<T> {
    /// Absolute opcode number
    pub opcode: T,
}

impl<T> VmAdapterInterface<T> for Rv32MultAdapterInterface<T> {
    type Reads = [[T; RV32_REGISTER_NUM_LANES]; 2];
    type Writes = [T; RV32_REGISTER_NUM_LANES];
    type ProcessedInstruction = Rv32MultProcessedInstruction<T>;
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32MultAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_index: T,
    pub rs2_index: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LANES>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32MultAdapterAir {
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32MultAdapterAir {
    fn width(&self) -> usize {
        size_of::<Rv32MultAdapterCols<u8>>()
    }
}

impl<AB: InteractionBuilder> Air<AB> for Rv32MultAdapterAir {
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32MultAdapterAir {
    type Interface = Rv32MultAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        todo!()
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32MultAdapter<F> {
    type ReadRecord = Rv32MultReadRecord<F>;
    type WriteRecord = Rv32MultWriteRecord<F>;
    type Air = Rv32MultAdapterAir;
    type Interface = Rv32MultAdapterInterface<F>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_b: b,
            op_c: c,
            d,
            ..
        } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), 1);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LANES>(d, b);
        let rs2 = memory.read::<RV32_REGISTER_NUM_LANES>(d, c);

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
        // TODO: timestamp delta debug check

        let Instruction { op_a: a, d, .. } = *instruction;
        let rd = memory.write(d, a, output.writes);

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
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
        todo!();
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
