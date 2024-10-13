use std::{marker::PhantomData, mem::size_of};

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        ExecutionState, InstructionOutput, CoreInterface, VmAdapter, VmAdapterAir,
        VmAdapterInterface, Result,
    },
    memory::{MemoryChip, MemoryWriteRecord},
    program::Instruction,
};

// This adapter doesn't read anything, and writes to [a:4]_d, where d == 1
#[derive(Debug, Clone, Default)]
pub struct Rv32RdWriteAdapter<F: Field> {
    _marker: PhantomData<F>,
    pub air: Rv32RdWriteAdapterAir,
}

impl<F: PrimeField32> Rv32RdWriteAdapter<F> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
            air: Rv32RdWriteAdapterAir {},
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32RdWriteWriteRecord<F: Field> {
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug, Clone)]
pub struct Rv32RdWriteProcessedInstruction<T> {
    pub _marker: PhantomData<T>,
}

pub struct Rv32RdWriteAdapterInterface<T>(PhantomData<T>);
impl<T: AbstractField> VmAdapterInterface<T> for Rv32RdWriteAdapterInterface<T> {
    type Reads = ();
    type Writes = [T; RV32_REGISTER_NUM_LANES];
    type ProcessedInstruction = Rv32RdWriteProcessedInstruction<T>;
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Rv32RdWriteAdapterCols<T> {
    pub _marker: PhantomData<T>,
}

impl<T> Rv32RdWriteAdapterCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32RdWriteAdapterCols<u8>>()
    }
}

#[derive(Clone, Copy, Debug, Default, derive_new::new)]
pub struct Rv32RdWriteAdapterAir {}

impl<F: Field> BaseAir<F> for Rv32RdWriteAdapterAir {
    fn width(&self) -> usize {
        size_of::<Rv32RdWriteAdapterCols<u8>>()
    }
}

impl<AB: InteractionBuilder> Air<AB> for Rv32RdWriteAdapterAir {
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32RdWriteAdapterAir {
    type Interface = Rv32RdWriteAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: CoreInterface<AB::Expr, Self::Interface>,
    ) {
        todo!()
    }
}

impl<F: PrimeField32> VmAdapter<F> for Rv32RdWriteAdapter<F> {
    type ReadRecord = ();
    type WriteRecord = Rv32RdWriteWriteRecord<F>;
    type Air = Rv32RdWriteAdapterAir;
    type Interface<T: AbstractField> = Rv32RdWriteAdapterInterface<T>;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let d = instruction.d;
        debug_assert_eq!(d.as_canonical_u32(), 1);

        Ok(((), ()))
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
