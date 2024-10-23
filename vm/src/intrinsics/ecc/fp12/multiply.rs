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
use axvm_instructions::FP12Opcode;
use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray,
        MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    intrinsics::ecc::Fp12BigUint,
    system::program::Instruction,
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
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
        BaseAir::<F>::width(&self.expr)
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
        assert_eq!(vars.len(), 34);
        assert_eq!(flags.len(), 1);
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = vars[vars.len() - 12..]
            .iter()
            .map(|x| (*x).into())
            .collect();

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

pub struct Fp12MultiplyCoreRecord {
    pub x: Fp12BigUint,
    pub y: Fp12BigUint,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for Fp12MultiplyCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    I::Writes: From<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = Fp12MultiplyCoreRecord;
    type Air = Fp12MultiplyCoreAir;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let num_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        // let Instruction { opcode, .. } = instruction.clone();
        // let local_opcode_index = opcode - self.air.offset;
        // let local_opcode = FP12Opcode::from_usize(local_opcode_index);
        // let mul_flag = match local_opcode {
        //     FP12Opcode::MUL => true,
        //     _ => panic!("Unsupported opcode: {:?}", local_opcode),
        // };

        let data: DynArray<_> = reads.into();
        let data = data.0;
        let x = data[..num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let y = data[..num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let x_biguint = limbs_to_biguint(&x, limb_bits);
        let y_biguint = limbs_to_biguint(&y, limb_bits);

        let vars = self
            .air
            .expr
            .execute(vec![x_biguint.clone(), y_biguint.clone()], vec![]);
        assert_eq!(vars.len(), 34);
        let res_biguint = vars[0].clone();
        tracing::trace!("FP12MultiplyOpcode | {res_biguint:?} | {x_biguint:?} | {y_biguint:?}");
        let res_limbs = biguint_to_limbs_vec(res_biguint, limb_bits, num_limbs);
        let writes = res_limbs
            .into_iter()
            .map(F::from_canonical_u32)
            .collect_vec();
        let ctx = AdapterRuntimeContext::<_, DynAdapterInterface<_>>::without_pc(writes);

        Ok((ctx.into(),))
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
