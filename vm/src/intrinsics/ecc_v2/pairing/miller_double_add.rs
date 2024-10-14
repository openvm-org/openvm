use afs_stark_backend::rap::BaseAirWithPublicValues;
use p3_air::{AirBuilder, BaseAir};
use p3_field::PrimeField32;

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, FlatInterface, Result, VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[derive(Clone)]
pub struct MillerDoubleAddAir {}

impl<F> BaseAir<F> for MillerDoubleAddAir {
    fn width(&self) -> usize {
        1 // TODO
    }
}

impl<F> BaseAirWithPublicValues<F> for MillerDoubleAddAir {}

impl<AB: AirBuilder, const READ_BYTES: usize, const WRITE_BYTES: usize>
    VmCoreAir<AB, FlatInterface<AB::Expr, READ_BYTES, WRITE_BYTES>> for MillerDoubleAddAir
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, FlatInterface<AB::Expr, READ_BYTES, WRITE_BYTES>> {
        todo!()
    }
}
pub struct MillerDoubleAddChip {
    pub air: MillerDoubleAddAir,
}

impl<F: PrimeField32, const READ_BYTES: usize, const WRITE_BYTES: usize>
    VmCoreChip<F, FlatInterface<F, READ_BYTES, WRITE_BYTES>> for MillerDoubleAddChip
{
    type Record = ();
    type Air = MillerDoubleAddAir;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: F,
        _reads: [F; READ_BYTES],
    ) -> Result<(
        AdapterRuntimeContext<F, FlatInterface<F, READ_BYTES, WRITE_BYTES>>,
        Self::Record,
    )> {
        todo!()
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "MillerDoubleAndAdd".to_string()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
