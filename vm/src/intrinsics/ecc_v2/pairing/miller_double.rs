use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr, FieldVariableConfig},
    field_extension::Fp2,
};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::PrimeField32;

use super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[derive(Clone)]
pub struct MillerDoubleAir<C: FieldVariableConfig> {
    pub expr: FieldExpr,
    pub _marker: PhantomData<C>,
}

impl<C: FieldVariableConfig> MillerDoubleAir<C> {
    pub fn new(modulus: BigUint, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        let limb_size = C::canonical_limb_bits();
        let num_limbs = C::num_limbs_per_field_element();
        assert!(modulus.bits() <= num_limbs * limb_size);
        let bus = range_checker.bus();
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            limb_size,
            bus.index,
            bus.range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        let builder = ExprBuilder::new(
            modulus,
            limb_size,
            num_limbs,
            range_checker.range_max_bits(),
        );
        let builder = Rc::new(RefCell::new(builder));

        let x = Fp2::<C>::new(builder.clone());
        let y = Fp2::<C>::new(builder.clone());
        // TODO:
        // λ = (3x^2) / (2y)
        // x_2S = λ^2 - 2x
        // y_2S = λ(x - x_2S) - y

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus: bus.index,
            range_max_bits: bus.range_max_bits,
        };
        Self {
            expr,
            _marker: PhantomData,
        }
    }
}

impl<F, C: FieldVariableConfig> BaseAir<F> for MillerDoubleAir<C> {
    fn width(&self) -> usize {
        1 // TODO
    }
}

impl<F, C: FieldVariableConfig> BaseAirWithPublicValues<F> for MillerDoubleAir<C> {}

impl<AB: AirBuilder, I, C: FieldVariableConfig> VmCoreAir<AB, I> for MillerDoubleAir<C>
where
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<Vec<AB::Expr>>,
    I::Writes: From<Vec<AB::Expr>>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        todo!()
    }
}

pub struct MillerDoubleChip<C: FieldVariableConfig> {
    pub air: MillerDoubleAir<C>,
}

impl<F: PrimeField32, I, C: FieldVariableConfig> VmCoreChip<F, I> for MillerDoubleChip<C>
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ();
    type Air = MillerDoubleAir<C>;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        // Input: EcPoint<Fp2>, so total 4 field elements.

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
