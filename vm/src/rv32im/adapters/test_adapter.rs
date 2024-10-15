use std::{fmt::Debug, marker::PhantomData};

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionState, Result, VmAdapterAir,
        VmAdapterChip, VmAdapterInterface,
    },
    system::{memory::MemoryController, program::Instruction},
};

// Replaces A: VmAdapterChip while testing VmCoreChip functionality, as it has no
// constraints and thus cannot cause a failure.
#[derive(Clone, Debug)]
pub struct TestAdapterChip<F: Field, A: VmAdapterChip<F>>
where
    A::Interface: VmAdapterInterface<F>,
    <A::Interface as VmAdapterInterface<F>>::Reads: Clone + Debug,
{
    pub air: TestAdapterAir<F, A::Air>,
    // What the test adapter will pass to core chip after preprocess
    pub reads: <A::Interface as VmAdapterInterface<F>>::Reads,
    // Amount to increment PC by, 4 by default
    pub pc_inc: Option<u32>,
}

impl<F: Field, A: VmAdapterChip<F>> TestAdapterChip<F, A>
where
    A::Interface: VmAdapterInterface<F>,
    <A::Interface as VmAdapterInterface<F>>::Reads: Clone + Debug,
{
    pub fn new(reads: <A::Interface as VmAdapterInterface<F>>::Reads, pc_inc: Option<u32>) -> Self {
        Self {
            air: TestAdapterAir {
                _marker_air: PhantomData,
                _marker_field: PhantomData,
            },
            reads,
            pc_inc,
        }
    }
}

impl<F: PrimeField32, A: VmAdapterChip<F>> VmAdapterChip<F> for TestAdapterChip<F, A>
where
    A::Interface: VmAdapterInterface<F>,
    <A::Interface as VmAdapterInterface<F>>::Reads: Clone + Debug,
{
    type ReadRecord = ();
    type WriteRecord = ();
    type Air = TestAdapterAir<F, A::Air>;
    type Interface = A::Interface;

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

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct TestAdapterAir<F: Field, B: BaseAir<F>> {
    _marker_field: PhantomData<F>,
    _marker_air: PhantomData<B>,
}

impl<F: Field, B: BaseAir<F>> BaseAir<F> for TestAdapterAir<F, B> {
    fn width(&self) -> usize {
        0
    }
}

impl<AB: InteractionBuilder, B: BaseAir<AB::F>> VmAdapterAir<AB> for TestAdapterAir<AB::F, B>
where
    B: VmAdapterAir<AB>,
{
    type Interface = B::Interface;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
    }
}
