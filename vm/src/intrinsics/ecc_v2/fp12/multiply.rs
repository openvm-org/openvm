use std::{marker::PhantomData, sync::Arc};

use afs_primitives::var_range::VariableRangeCheckerChip;
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ark_ff::{Fp2, Fp2Config};
use ax_ecc_primitives::field_expression::{FieldExpr, FieldVariableConfig};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::PrimeField32;

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, FlatInterface, Result, VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[derive(Clone)]
pub struct Fp12MultiplyAir<C: FieldVariableConfig> {
    pub expr: FieldExpr,
    _marker: PhantomData<C>,
}

impl<C: FieldVariableConfig, F: Fp2Config> Fp12MultiplyAir<C> {
    pub fn new(
        modulus: BigUint,
        offset: usize,
        xi: &Fp2<F>,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker,
        };
        Self { expr, offset }
    }
}

impl<F, C: FieldVariableConfig> BaseAir<F> for Fp12MultiplyAir<C> {
    fn width(&self) -> usize {
        // BaseAir::<F>::width(&self.expr)
        todo!()
    }
}

impl<F, C: FieldVariableConfig> BaseAirWithPublicValues<F> for Fp12MultiplyAir<C> {}

impl<AB: AirBuilder, const READ_BYTES: usize, const WRITE_BYTES: usize, C: FieldVariableConfig>
    VmCoreAir<AB, FlatInterface<AB::Expr, READ_BYTES, WRITE_BYTES>> for Fp12MultiplyAir<C>
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

pub struct Fp12MultiplyChip<C: FieldVariableConfig> {
    pub air: Fp12MultiplyAir<C>,
}

impl<
        F: PrimeField32,
        const READ_BYTES: usize,
        const WRITE_BYTES: usize,
        C: FieldVariableConfig,
    > VmCoreChip<F, FlatInterface<F, READ_BYTES, WRITE_BYTES>> for Fp12MultiplyChip<C>
{
    type Record = ();
    type Air = Fp12MultiplyAir<C>;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: [F; READ_BYTES],
    ) -> Result<(
        AdapterRuntimeContext<F, FlatInterface<F, READ_BYTES, WRITE_BYTES>>,
        Self::Record,
    )> {
        // Input: EcPoint<Fp2>, so total 4 field elements.
        let field_element_limbs = C::num_limbs_per_field_element();
        assert_eq!(reads.len(), 4 * field_element_limbs);

        todo!()
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "Fp12Multiply".to_string()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
