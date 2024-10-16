use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::rap::BaseAirWithPublicValues;
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExpr, FieldVariable, SymbolicExpr};
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{Field, PrimeField32};

use super::{ModularConfig, FIELD_ELEMENT_BITS};
use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::Rv32HeapAdapterInterface,
    system::program::Instruction,
    utils::{biguint_to_limbs, limbs_to_biguint},
};

#[derive(Clone)]
pub struct ModularMulDivV2CoreAir<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub expr: FieldExpr,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE> {
    pub fn new(modulus: BigUint, range_bus: usize, range_max_bits: usize) -> Self {
        assert!(modulus.bits() <= NUM_LIMBS * LIMB_SIZE);
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            LIMB_SIZE,
            range_bus,
            range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        let builder = ExprBuilder::new(modulus, LIMB_SIZE, NUM_LIMBS, range_max_bits);
        let builder = Rc::new(RefCell::new(builder));
        let x = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let y = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let z = builder.borrow_mut().new_var();
        let z = FieldVariable::from_var(builder.clone(), z);
        let is_mul_flag = builder.borrow_mut().new_flag();
        // constraint is x * y = z, or z * y = x
        let lvar = FieldVariable::select(is_mul_flag, &x, &z);
        let rvar = FieldVariable::select(is_mul_flag, &z, &x);
        let constraint = lvar * y.clone() - rvar;
        builder.borrow_mut().add_constraint(constraint.expr);
        let compute = SymbolicExpr::Div(Box::new(x.expr), Box::new(y.expr));
        builder.borrow_mut().add_compute(compute);

        let builder = builder.borrow().clone();

        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
            range_max_bits,
        };
        Self { expr }
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_SIZE: usize> BaseAir<F>
    for ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>
{
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_SIZE: usize> BaseAirWithPublicValues<F>
    for ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>
{
}

impl<AB: AirBuilder, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    VmCoreAir<AB, Rv32HeapAdapterInterface<AB::Expr, NUM_LIMBS, NUM_LIMBS>>
    for ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, Rv32HeapAdapterInterface<AB::Expr, NUM_LIMBS, NUM_LIMBS>> {
        todo!()
    }
}

#[derive(Clone)]
pub struct ModularMulDivV2CoreChip<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub air: ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>,
    pub offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularMulDivV2CoreChip<NUM_LIMBS, LIMB_SIZE> {
    pub fn new(
        modulus: BigUint,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let air = ModularMulDivV2CoreAir::new(
            modulus,
            range_checker.bus().index,
            range_checker.range_max_bits(),
        );
        Self { air, offset }
    }
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    VmCoreChip<F, Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS>>
    for ModularMulDivV2CoreChip<NUM_LIMBS, LIMB_SIZE>
{
    type Record = ();
    type Air = ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: <Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(
        AdapterRuntimeContext<F, Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS>>,
        Self::Record,
    )> {
        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_index = opcode - self.offset;
        let (x, y) = reads;
        let x = x.map(|x| x.as_canonical_u32());
        let y = y.map(|x| x.as_canonical_u32());

        let x_biguint = limbs_to_biguint(&x, LIMB_SIZE);
        let y_biguint = limbs_to_biguint(&y, LIMB_SIZE);

        let opcode = ModularArithmeticOpcode::from_usize(local_opcode_index);
        let is_mul_flag = match opcode {
            ModularArithmeticOpcode::MUL => true,
            ModularArithmeticOpcode::DIV => false,
            _ => panic!("Unsupported opcode: {:?}", opcode),
        };

        let vars = self
            .air
            .expr
            .execute(vec![x_biguint, y_biguint], vec![is_mul_flag]);
        assert_eq!(vars.len(), 1);
        let z_biguint = vars[0].clone();
        let z_limbs = biguint_to_limbs::<NUM_LIMBS>(z_biguint, LIMB_SIZE);

        Ok((
            AdapterRuntimeContext {
                to_pc: None,
                writes: z_limbs.map(|x| F::from_canonical_u32(x)),
            },
            (),
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "todo".to_string()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
