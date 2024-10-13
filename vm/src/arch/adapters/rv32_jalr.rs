use std::{marker::PhantomData, mem::size_of};

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        ExecutionState, InstructionOutput, IntegrationInterface, MachineAdapter,
        MachineAdapterInterface, Result,
    },
    memory::{MemoryChip, MemoryReadRecord, MemoryWriteRecord},
    program::Instruction,
};

// This adapter reads from [b:4]_d (rs1) and writes to [a:4]_d (rd)
#[derive(Debug, Clone, Default)]
pub struct Rv32JalrAdapter<F: Field> {
    _marker: PhantomData<F>,
    pub air: Rv32JalrAdapterAir,
}

impl<F: PrimeField32> Rv32JalrAdapter<F> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
            air: Rv32JalrAdapterAir {},
        }
    }
}
#[derive(Debug, Clone)]
pub struct Rv32JalrReadRecord<F: Field> {
    pub rs1: MemoryReadRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug, Clone)]
pub struct Rv32JalrWriteRecord<F: Field> {
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug, Clone)]
pub struct Rv32JalrProcessedInstruction<T> {
    pub _marker: PhantomData<T>,
}

pub struct Rv32JalrAdapterInterface<T>(PhantomData<T>);
impl<T: AbstractField> MachineAdapterInterface<T> for Rv32JalrAdapterInterface<T> {
    type Reads = [T; RV32_REGISTER_NUM_LANES];
    type Writes = [T; RV32_REGISTER_NUM_LANES];
    type ProcessedInstruction = Rv32JalrProcessedInstruction<T>;
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Rv32JalrAdapterCols<T> {
    pub _marker: PhantomData<T>,
}

impl<T> Rv32JalrAdapterCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32JalrAdapterCols<u8>>()
    }
}

#[derive(Clone, Copy, Debug, Default, derive_new::new)]
pub struct Rv32JalrAdapterAir {}

impl<F: Field> BaseAir<F> for Rv32JalrAdapterAir {
    fn width(&self) -> usize {
        size_of::<Rv32JalrAdapterCols<u8>>()
    }
}

impl<AB: InteractionBuilder> Air<AB> for Rv32JalrAdapterAir {
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<F: PrimeField32> MachineAdapter<F> for Rv32JalrAdapter<F> {
    type ReadRecord = Rv32JalrReadRecord<F>;
    type WriteRecord = Rv32JalrWriteRecord<F>;
    type Air = Rv32JalrAdapterAir;
    type Interface<T: AbstractField> = Rv32JalrAdapterInterface<T>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as MachineAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { op_b: b, d, .. } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), 1);

        let rs1 = memory.read::<RV32_REGISTER_NUM_LANES>(d, b);

        Ok((rs1.data, Rv32JalrReadRecord { rs1 }))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: InstructionOutput<F, Self::Interface<F>>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        let Instruction { op_a: a, d, .. } = *instruction;
        let rd = memory.write(d, a, output.writes);

        Ok((
            ExecutionState {
                pc: output
                    .to_pc
                    .unwrap_or(F::from_canonical_usize(from_state.pc + 4))
                    .as_canonical_u32() as usize,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            Self::WriteRecord { rd },
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
