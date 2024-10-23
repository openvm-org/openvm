use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    sub_chip::{LocalTraceInstructions, SubAir},
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExpr, FieldExprCols};
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::super::{EcPoint, FIELD_ELEMENT_BITS};
use crate::{
    arch::{
        instructions::EccOpcode, AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface,
        DynArray, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
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
        modulus: BigUint, // The coordinate field.
        num_limbs: usize,
        limb_bits: usize,
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
        let builder = ExprBuilder::new(modulus, limb_bits, num_limbs, range_bus.range_max_bits);
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

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for SwEcAddNeCoreAir
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
        assert_eq!(inputs.len(), 4);
        assert_eq!(vars.len(), 3);
        assert_eq!(flags.len(), 0);
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = vars[1..].concat().iter().map(|x| (*x).into()).collect();

        let expected_opcode = EccOpcode::EC_ADD_NE as usize;
        let expected_opcode = AB::Expr::from_canonical_usize(expected_opcode);

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: expected_opcode + AB::Expr::from_canonical_usize(self.offset),
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

pub struct SwEcAddNeCoreChip {
    pub air: SwEcAddNeCoreAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl SwEcAddNeCoreChip {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let air = SwEcAddNeCoreAir::new(modulus, num_limbs, limb_bits, range_checker.bus(), offset);
        Self { air, range_checker }
    }
}

pub struct SwEcAddNeCoreRecord {
    pub p1: EcPoint,
    pub p2: EcPoint,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for SwEcAddNeCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = SwEcAddNeCoreRecord;
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
        let data: DynArray<_> = reads.into();
        let data = data.0;
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

        let vars = self
            .air
            .expr
            .execute(vec![x1.clone(), y1.clone(), x2.clone(), y2.clone()], vec![]);
        assert_eq!(vars.len(), 3); // lambda, x3, y3
        let x3 = vars[1].clone();
        let y3 = vars[2].clone();

        let x3_limbs = biguint_to_limbs_vec(x3, limb_bits, field_element_limbs);
        let y3_limbs = biguint_to_limbs_vec(y3, limb_bits, field_element_limbs);
        let writes = [x3_limbs, y3_limbs]
            .concat()
            .into_iter()
            .map(|x| F::from_canonical_u32(x))
            .collect::<Vec<_>>();
        let ctx = AdapterRuntimeContext::<_, DynAdapterInterface<_>>::without_pc(writes);

        Ok((
            ctx.into(),
            SwEcAddNeCoreRecord {
                p1: EcPoint { x: x1, y: y1 },
                p2: EcPoint { x: x2, y: y2 },
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "SwEcAddNe".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let input = (
            vec![record.p1.x, record.p1.y, record.p2.x, record.p2.y],
            self.range_checker.clone(),
            vec![],
        );
        let row = LocalTraceInstructions::<F>::generate_trace_row(&self.air.expr, input);
        for (i, element) in row.iter().enumerate() {
            row_slice[i] = *element;
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
