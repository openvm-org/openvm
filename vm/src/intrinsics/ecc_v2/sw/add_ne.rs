use std::{cell::RefCell, rc::Rc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    var_range::bus::VariableRangeCheckerBus,
};
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExpr};
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
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
};

#[derive(Clone)]
pub struct SwEcAddNeCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl SwEcAddNeCoreAir {
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

        let x1 = ExprBuilder::new_input(builder.clone());
        let y1 = ExprBuilder::new_input(builder.clone());
        let x2 = ExprBuilder::new_input(builder.clone());
        let y2 = ExprBuilder::new_input(builder.clone());
        let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
        let mut x3 = lambda.square() - x1.clone() - x2;
        x3.save();
        let mut y3 = lambda * (x1 - x3.clone()) - y1;
        y3.save();

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
        };
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for SwEcAddNeCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for SwEcAddNeCoreAir {}

impl<AB: AirBuilder, I> VmCoreAir<AB, I> for SwEcAddNeCoreAir
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

pub struct SwEcAddNeCoreChip {
    pub air: SwEcAddNeCoreAir,
}

impl SwEcAddNeCoreChip {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        max_limb_bits: usize,
        range_bus: VariableRangeCheckerBus,
        offset: usize,
    ) -> Self {
        let air = SwEcAddNeCoreAir::new(
            modulus,
            num_limbs,
            limb_bits,
            max_limb_bits,
            range_bus,
            offset,
        );
        Self { air }
    }
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for SwEcAddNeCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ();
    type Air = SwEcAddNeCoreAir;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        // Input: 2 EcPoint<Fp>, so total 4 field elements.
        let field_element_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        let data: Vec<F> = reads.into();
        assert_eq!(data.len(), 4 * field_element_limbs);
        let data_u32: Vec<u32> = data.iter().map(|x| x.as_canonical_u32()).collect();

        let x1 = limbs_to_biguint(&data_u32[..field_element_limbs], limb_bits);
        let y1 = limbs_to_biguint(
            &data_u32[field_element_limbs..2 * field_element_limbs],
            limb_bits,
        );
        let x2 = limbs_to_biguint(
            &data_u32[2 * field_element_limbs..3 * field_element_limbs],
            limb_bits,
        );
        let y2 = limbs_to_biguint(
            &data_u32[3 * field_element_limbs..4 * field_element_limbs],
            limb_bits,
        );

        let vars = self.air.expr.execute(vec![x1, y1, x2, y2], vec![]);
        assert_eq!(vars.len(), 3); // lambda, x3, y3
        let x3 = vars[1].clone();
        let y3 = vars[2].clone();

        let x3_limbs = biguint_to_limbs_vec(x3, limb_bits, field_element_limbs);
        let y3_limbs = biguint_to_limbs_vec(y3, limb_bits, field_element_limbs);

        Ok((
            AdapterRuntimeContext {
                to_pc: None,
                writes: [x3_limbs, y3_limbs]
                    .concat()
                    .into_iter()
                    .map(|x| F::from_canonical_u32(x))
                    .collect::<Vec<_>>()
                    .into(),
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
