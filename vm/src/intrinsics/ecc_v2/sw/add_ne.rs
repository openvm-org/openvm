use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExpr, FieldVariableConfig};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{Field, PrimeField32};

use super::super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, FlatInterface, Result, VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
    utils::{biguint_to_limbs, limbs_to_biguint},
};

#[derive(Clone)]
pub struct SwEcAddNeAir<C: FieldVariableConfig> {
    pub expr: FieldExpr,
    pub _marker: PhantomData<C>,
}

impl<C: FieldVariableConfig> SwEcAddNeAir<C> {
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

        let x1 = ExprBuilder::new_input::<C>(builder.clone());
        let y1 = ExprBuilder::new_input::<C>(builder.clone());
        let x2 = ExprBuilder::new_input::<C>(builder.clone());
        let y2 = ExprBuilder::new_input::<C>(builder.clone());
        let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
        let mut x3 = lambda.square() - x1.clone() - x2;
        x3.save();
        let mut y3 = lambda * (x1 - x3.clone()) - y1;
        y3.save();

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker,
        };
        Self {
            expr,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, C: FieldVariableConfig> BaseAir<F> for SwEcAddNeAir<C> {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field, C: FieldVariableConfig> BaseAirWithPublicValues<F> for SwEcAddNeAir<C> {}

impl<AB: AirBuilder, C: FieldVariableConfig, const READ_BYTES: usize, const WRITE_BYTES: usize>
    VmCoreAir<AB, FlatInterface<AB::Expr, READ_BYTES, WRITE_BYTES>> for SwEcAddNeAir<C>
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

pub struct SwEcAddNeChip<C: FieldVariableConfig> {
    pub air: SwEcAddNeAir<C>,
}

impl<
        F: PrimeField32,
        const READ_BYTES: usize,
        const WRITE_BYTES: usize,
        C: FieldVariableConfig,
    > VmCoreChip<F, FlatInterface<F, READ_BYTES, WRITE_BYTES>> for SwEcAddNeChip<C>
{
    type Record = ();
    type Air = SwEcAddNeAir<C>;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: [F; READ_BYTES],
    ) -> Result<(
        AdapterRuntimeContext<F, FlatInterface<F, READ_BYTES, WRITE_BYTES>>,
        Self::Record,
    )> {
        // Input: 2 EcPoint<Fp>, so total 4 field elements.
        let field_element_limbs = C::num_limbs_per_field_element();
        let limb_bits = C::canonical_limb_bits();
        assert_eq!(reads.len(), 4 * field_element_limbs);
        let x1 = reads[0..field_element_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        let y1 = reads[field_element_limbs..2 * field_element_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        let x2 = reads[2 * field_element_limbs..3 * field_element_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        let y2 = reads[3 * field_element_limbs..4 * field_element_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();

        let x1 = limbs_to_biguint(&x1, limb_bits);
        let y1 = limbs_to_biguint(&y1, limb_bits);
        let x2 = limbs_to_biguint(&x2, limb_bits);
        let y2 = limbs_to_biguint(&y2, limb_bits);

        let vars = self.air.expr.execute(vec![x1, y1, x2, y2], vec![]);
        assert_eq!(vars.len(), 3); // lambda, x3, y3
        let x3 = vars[1].clone();
        let y3 = vars[2].clone();

        // FIX this: should to_limbs not take a const generic?
        let x3_limbs = biguint_to_limbs::<field_element_limbs>(x3, limb_bits);
        let y3_limbs = biguint_to_limbs::<field_element_limbs>(y3, limb_bits);

        Ok((
            AdapterRuntimeContext {
                to_pc: None,
                writes: [x3_limbs, y3_limbs]
                    .concat()
                    .into_iter()
                    .map(|x| F::from_canonical_u32(x))
                    .try_into()
                    .unwrap(),
            },
            (),
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "SwEcAddNe".to_string()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
