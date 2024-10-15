use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use super::Rv32RTypeAdapterInterface;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionState, Result, VmAdapterAir,
        VmAdapterChip, VmAdapterInterface,
    },
    system::{memory::MemoryController, program::Instruction},
};

#[derive(Clone, Debug)]
pub struct Rv32TestAdapterChip<T, I: VmAdapterInterface<T>>
where
    I::Reads: Clone,
{
    pub air: Rv32TestAdapterAir,
    // What the test adapter will pass to core chip after preprocess
    pub reads: I::Reads,
    // Amount to increment PC by, 4 by default
    pub pc_inc: Option<u32>,
}

impl<T, I: VmAdapterInterface<T>> Rv32TestAdapterChip<T, I>
where
    I::Reads: Clone,
{
    pub fn new(reads: I::Reads, pc_inc: Option<u32>) -> Self {
        Self {
            air: Rv32TestAdapterAir {},
            reads,
            pc_inc,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32TestAdapterAir {}

impl<F: Field> BaseAir<F> for Rv32TestAdapterAir {
    fn width(&self) -> usize {
        0
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32TestAdapterAir {
    type Interface = Rv32RTypeAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F> + Clone> VmAdapterChip<F>
    for Rv32TestAdapterChip<F, I>
where
    I::Reads: Clone,
{
    type ReadRecord = ();
    type WriteRecord = ();
    type Air = Rv32TestAdapterAir;
    type Interface = I;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        Ok((self.reads.clone(), ()))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        _output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        Ok((
            ExecutionState {
                pc: from_state.pc + self.pc_inc.unwrap_or(4),
                timestamp: memory.timestamp(),
            },
            (),
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
