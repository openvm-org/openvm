use std::{array, marker::PhantomData, mem::size_of};

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{Rv32AuipcOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapter, VmAdapterInterface, VmCore,
        VmCoreAir, Writes, RV32_REGISTER_NUM_LANES,
    },
    program::Instruction,
};

#[derive(Debug, Clone)]
pub struct Rv32AuipcCols<T> {
    pub _marker: PhantomData<T>,
}

impl<T> Rv32AuipcCols<T> {
    pub fn width() -> usize {
        size_of::<Rv32AuipcCols<T>>()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcAir<F: Field> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32AuipcAir<F> {
    fn width(&self) -> usize {
        Rv32AuipcCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32AuipcAir<F> {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for Rv32AuipcAir<AB::F>
where
    I: VmAdapterInterface<AB::Expr>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Rv32AuipcCore<F: Field> {
    pub air: Rv32AuipcAir<F>,
}

impl<F: Field> Rv32AuipcCore<F> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: Rv32AuipcAir::<F> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, A: VmAdapterChip<F>> VmCoreChip<F, A> for Rv32AuipcCore<F>
where
    Writes<F, A::Interface<F>>: From<[F; RV32_REGISTER_NUM_LANES]>,
{
    type Record = ();
    type Air = Rv32AuipcAir<F>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        _reads: <A::Interface<F> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(AdapterRuntimeContext<F, A::Interface<F>>, Self::Record)> {
        let opcode = Rv32AuipcOpcode::from_usize(instruction.opcode - self.air.offset);
        let c = instruction.op_c.as_canonical_u32();
        let rd_data = solve_auipc(opcode, from_pc.as_canonical_u32(), c);
        let rd_data = rd_data.map(F::from_canonical_u32);

        let output: AdapterRuntimeContext<F, A::Interface<F>> = AdapterRuntimeContext {
            to_pc: None,
            writes: rd_data.into(),
        };

        // TODO: send XorLookUpChip requests
        // TODO: create Record and return

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32AuipcOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns rd_data
pub(super) fn solve_auipc(
    _opcode: Rv32AuipcOpcode,
    pc: u32,
    imm: u32,
) -> [u32; RV32_REGISTER_NUM_LANES] {
    let rd = pc.wrapping_add(imm << 8);
    array::from_fn(|i| (rd >> (8 * i)) & 255)
}
