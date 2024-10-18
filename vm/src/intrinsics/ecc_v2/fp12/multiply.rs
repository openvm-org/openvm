use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr},
    field_extension::Fp2,
};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{Field, PrimeField32};

use super::super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[derive(Clone)]
pub struct Fp12MultiplyAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl Fp12MultiplyAir {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
        max_limb_bits: usize,
        offset: usize,
        xi: Fp2,
    ) -> Self {
        assert!(modulus.bits() <= num_limbs * limb_bits);
        let range_bus = range_checker.bus();
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            limb_bits,
            range_bus.index,
            range_bus.range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        let builder = ExprBuilder::new(
            modulus,
            limb_bits,
            num_limbs,
            range_checker.range_max_bits(),
            max_limb_bits,
        );
        let builder = Rc::new(RefCell::new(builder));

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
        };
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for Fp12MultiplyAir {
    fn width(&self) -> usize {
        // BaseAir::<F>::width(&self.expr)
        todo!()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Fp12MultiplyAir {}

impl<AB: AirBuilder, I> VmCoreAir<AB, I> for Fp12MultiplyAir
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

pub struct Fp12MultiplyChip {
    pub air: Fp12MultiplyAir,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for Fp12MultiplyChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ();
    type Air = Fp12MultiplyAir;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
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
