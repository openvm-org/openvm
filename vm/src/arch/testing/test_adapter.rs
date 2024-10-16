use std::{borrow::Borrow, collections::VecDeque, fmt::Debug};

use afs_derive::AlignedBorrow;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray, ExecutionState,
        Result, VmAdapterAir, VmAdapterChip,
    },
    system::{memory::MemoryController, program::Instruction},
};

// Replaces A: VmAdapterChip while testing VmCoreChip functionality, as it has no
// constraints and thus cannot cause a failure.
#[derive(Clone, Debug)]
pub struct TestAdapterChip<F> {
    /// List of the return values of `preprocess` this chip should provide on each sequential call.
    pub prank_reads: VecDeque<Vec<F>>,
    /// List of `pc_inc` to use in `postprocess` on each sequential call.
    /// Defaults to `4` if not provided.
    pub prank_pc_inc: VecDeque<Option<u32>>,
}

impl<F> TestAdapterChip<F> {
    pub fn new(prank_reads: Vec<Vec<F>>, prank_pc_inc: Vec<Option<u32>>) -> Self {
        Self {
            prank_reads: prank_reads.into(),
            prank_pc_inc: prank_pc_inc.into(),
        }
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for TestAdapterChip<F> {
    type ReadRecord = ();
    type WriteRecord = ();
    type Air = EmptyAir;
    type Interface = DynAdapterInterface<F>;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
    ) -> Result<(DynArray<F>, Self::ReadRecord)> {
        Ok((
            self.prank_reads
                .pop_front()
                .expect("Not enough prank reads provided")
                .into(),
            (),
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        _output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let pc_inc = self
            .prank_pc_inc
            .pop_front()
            .map(|x| x.unwrap_or(4))
            .unwrap_or(4);
        Ok((
            ExecutionState {
                pc: from_state.pc + pc_inc,
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
        &EmptyAir
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EmptyAir;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct EmptyAirCols<T> {
    pub dummy: T,
}

impl<F: Field> BaseAir<F> for EmptyAir {
    fn width(&self) -> usize {
        1
    }
}

impl<AB: AirBuilder> VmAdapterAir<AB> for EmptyAir {
    type Interface = DynAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        _builder: &mut AB,
        local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let _cols: &EmptyAirCols<_> = local.borrow();
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        // TODO: This is a hack to make the code compile, as it is not used anywhere
        local[0]
    }
}
