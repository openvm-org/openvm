use std::{marker::PhantomData, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        Result, VmAdapter, VmAdapterAir, VmAdapterInterface,
    },
    memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols},
        MemoryChip, MemoryChipRef, MemoryReadRecord,
    },
    program::{bridge::ProgramBus, Instruction},
};

/// Reads instructions of the form OP a, b, c, d, e where if([a:4]_d op [b:4]_e) pc += c.
/// Operands d and e can only be 1.
#[derive(Debug)]
pub struct Rv32BranchAdapter<F: Field> {
    _marker: PhantomData<F>,
    pub air: Rv32BranchAdapterAir,
}

impl<F: PrimeField32> Rv32BranchAdapter<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            _marker: PhantomData,
            air: Rv32BranchAdapterAir {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _memory_bridge: memory_bridge,
            },
        }
    }
}

#[derive(Debug)]
pub struct Rv32BranchReadRecord<F: Field> {
    /// Read register value from address space d=1
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
    /// Read register value from address space e=1
    pub rs2: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug)]
pub struct Rv32BranchWriteRecord {
    pub from_state: ExecutionState<usize>,
}

pub struct Rv32BranchAdapterInterface<T>(PhantomData<T>);

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BranchProcessedInstruction<T> {
    /// Absolute opcode number
    pub opcode: T,
    /// Amount to increment PC by (4 if branch condition failed)
    pub pc_inc: T,
}

impl<T: AbstractField> VmAdapterInterface<T> for Rv32BranchAdapterInterface<T> {
    type Reads = [[T; RV32_REGISTER_NUM_LANES]; 2];
    type Writes = ();
    type ProcessedInstruction = Rv32BranchProcessedInstruction<T>;
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BranchAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_index: T,
    pub rs2_index: T,
    pub imm: T,
    pub reads_aux: [MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BranchAdapterAir {
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32BranchAdapterAir {
    fn width(&self) -> usize {
        size_of::<Rv32BranchAdapterCols<u8>>()
    }
}

impl<AB: InteractionBuilder> Air<AB> for Rv32BranchAdapterAir {
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32BranchAdapterAir {
    type Interface = Rv32BranchAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        todo!()
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32BranchAdapter<F> {
    type ReadRecord = Rv32BranchReadRecord<F>;
    type WriteRecord = Rv32BranchWriteRecord;
    type Air = Rv32BranchAdapterAir;
    type Interface<T: AbstractField> = Rv32BranchAdapterInterface<T>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as VmAdapterInterface<F>>::Reads,
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
        memory: &mut MemoryChip<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: AdapterRuntimeContext<F, Self::Interface<F>>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        // TODO: timestamp delta debug check

        let to_pc = output
            .to_pc
            .map(|x| x.as_canonical_u32() as usize)
            .unwrap_or(from_state.pc + 4);

        Ok((
            ExecutionState {
                pc: to_pc,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            Self::WriteRecord { from_state },
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
