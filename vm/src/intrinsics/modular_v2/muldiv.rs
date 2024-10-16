use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    sub_chip::{LocalTraceInstructions, SubAir},
    var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use ax_ecc_primitives::field_expression::{
    ExprBuilder, FieldExpr, FieldExprCols, FieldVariable, SymbolicExpr,
};
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{ModularConfig, FIELD_ELEMENT_BITS};
use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
    utils::{biguint_to_limbs, limbs_to_biguint},
};

#[derive(Clone)]
pub struct ModularMulDivV2CoreAir<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE> {
    pub fn new(modulus: BigUint, range_bus: usize, range_max_bits: usize, offset: usize) -> Self {
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
        let compute = SymbolicExpr::Select(
            is_mul_flag,
            Box::new(x.expr.clone() * y.expr.clone()),
            Box::new(x.expr.clone() / y.expr.clone()),
        );
        builder.borrow_mut().add_compute(compute);

        let builder = builder.borrow().clone();

        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
            range_max_bits,
        };
        Self { expr, offset }
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

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_SIZE: usize, I> VmCoreAir<AB, I>
    for ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>
where
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<Vec<AB::Expr>>,
    I::Writes: From<Vec<AB::Expr>>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
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
        assert_eq!(inputs.len(), 2);
        assert_eq!(vars.len(), 1);
        assert_eq!(flags.len(), 1);
        let reads = inputs
            .concat()
            .iter()
            .map(|x| (*x).into())
            .collect::<Vec<_>>();
        let writes = vars[0].iter().map(|x| (*x).into()).collect::<Vec<_>>();
        // flag = 1 means mul (opcode = 2), flag = 0 means div (opcode = 3)
        let expected_opcode = AB::Expr::from_canonical_usize(3) - flags[0];

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: expected_opcode + AB::Expr::from_canonical_usize(self.offset),
        };

        AdapterAirContext {
            to_pc: None,
            reads: reads.into(),
            writes: writes.into(),
            instruction: instruction.into(),
        }
    }
}

#[derive(Clone)]
pub struct ModularMulDivV2CoreChip<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub air: ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>,
    pub range_checker: Arc<VariableRangeCheckerChip>,
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
            offset,
        );
        Self { air, range_checker }
    }
}

pub struct ModularMulDivV2CoreRecord {
    pub x: BigUint,
    pub y: BigUint,
    pub is_mul_flag: bool,
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize, I> VmCoreChip<F, I>
    for ModularMulDivV2CoreChip<NUM_LIMBS, LIMB_SIZE>
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<Vec<F>>,
    I::Writes: From<Vec<F>>,
{
    type Record = ModularMulDivV2CoreRecord;
    type Air = ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_SIZE>;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_index = opcode - self.air.offset;
        let data: Vec<F> = reads.into();
        assert_eq!(data.len(), 2 * NUM_LIMBS);
        let x = data[..NUM_LIMBS]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();
        let y = data[NUM_LIMBS..]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect::<Vec<_>>();

        let x_biguint = limbs_to_biguint(&x, LIMB_SIZE);
        let y_biguint = limbs_to_biguint(&y, LIMB_SIZE);

        let opcode = ModularArithmeticOpcode::from_usize(local_opcode_index);
        let is_mul_flag = match opcode {
            ModularArithmeticOpcode::MUL => true,
            ModularArithmeticOpcode::DIV => false,
            _ => panic!("Unsupported opcode: {:?}", opcode),
        };

        let vars = self.air.expr.execute(
            vec![x_biguint.clone(), y_biguint.clone()],
            vec![is_mul_flag],
        );
        assert_eq!(vars.len(), 1);
        let z_biguint = vars[0].clone();
        let z_limbs = biguint_to_limbs::<NUM_LIMBS>(z_biguint, LIMB_SIZE);

        Ok((
            AdapterRuntimeContext {
                to_pc: None,
                writes: z_limbs.map(|x| F::from_canonical_u32(x)).to_vec().into(),
            },
            ModularMulDivV2CoreRecord {
                x: x_biguint,
                y: y_biguint,
                is_mul_flag,
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "ModularMulDiv".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let input = (
            vec![record.x, record.y],
            self.range_checker.clone(),
            vec![record.is_mul_flag],
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
