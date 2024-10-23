use std::{cell::RefCell, rc::Rc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    sub_chip::SubAir,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr, FieldExprCols},
    field_extension::{Fp12, Fp2},
};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, MinimalInstruction, Result,
        VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[derive(Clone)]
pub struct Fp12MultiplyCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl Fp12MultiplyCoreAir {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        max_limb_bits: usize,
        range_bus: VariableRangeCheckerBus,
        offset: usize,
    ) -> Self {
        assert!(modulus.bits() <= num_limbs * limb_bits);
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
            range_bus.range_max_bits,
            max_limb_bits,
        );
        let builder = Rc::new(RefCell::new(builder));

        let mut x = Fp12::new(builder.clone());
        let mut y = Fp12::new(builder.clone());
        let mut xi = Fp2::new(builder.clone());
        let mut res = x.mul(&mut y, &mut xi);
        res.save();

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
        };
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for Fp12MultiplyCoreAir {
    fn width(&self) -> usize {
        // BaseAir::<F>::width(&self.expr)
        todo!()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Fp12MultiplyCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for Fp12MultiplyCoreAir
where
    I: VmAdapterInterface<AB::Expr>,
    AdapterAirContext<AB::Expr, I>:
        From<AdapterAirContext<AB::Expr, DynAdapterInterface<AB::Expr>>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        assert_eq!(local.len(), BaseAir::<AB::F>::width(&self.expr));
        SubAir::eval(&self.expr, builder, local.to_vec(), ());

        let FieldExprCols {
            is_valid,
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local);
        assert_eq!(inputs.len(), 3);
        assert_eq!(vars.len(), 1);
        assert_eq!(flags.len(), 1);
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = vars[0].iter().map(|x| (*x).into()).collect();

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: AB::Expr::from_canonical_usize(self.offset),
        };

        let ctx: AdapterAirContext<_, DynAdapterInterface<_>> = AdapterAirContext {
            to_pc: None,
            reads: reads.into(),
            writes: writes.into(),
            instruction: instruction.into(),
        };
        ctx.into()
    }
}

pub struct Fp12MultiplyCoreChip {
    pub air: Fp12MultiplyCoreAir,
    pub xi: Fp2,
}

impl Fp12MultiplyCoreChip {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        max_limb_bits: usize,
        range_bus: VariableRangeCheckerBus,
        offset: usize,
        xi: Fp2,
    ) -> Self {
        let air = Fp12MultiplyCoreAir::new(
            modulus,
            num_limbs,
            limb_bits,
            max_limb_bits,
            range_bus,
            offset,
        );
        Self { air, xi }
    }
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for Fp12MultiplyCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ();
    type Air = Fp12MultiplyCoreAir;

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
