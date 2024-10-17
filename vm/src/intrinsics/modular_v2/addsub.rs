use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    sub_chip::{LocalTraceInstructions, SubAir},
    var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExpr, FieldExprCols, FieldVariable};
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
pub struct ModularAddSubV2CoreAir<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_SIZE> {
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
        let x1 = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let x2 = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let x3 = x1.clone() + x2.clone();
        let x4 = x1 - x2;
        let is_add_flag = builder.borrow_mut().new_flag();
        let mut x5 = FieldVariable::select(is_add_flag, &x3, &x4);
        x5.save();
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
    for ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_SIZE>
{
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_SIZE: usize> BaseAirWithPublicValues<F>
    for ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_SIZE>
{
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_SIZE: usize, I> VmCoreAir<AB, I>
    for ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_SIZE>
where
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[AB::Expr; NUM_LIMBS]>,
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
        // flag = 1 means add (opcode = 0), flag = 0 means sub (opcode = 1)
        let expected_opcode = AB::Expr::one() - flags[0];

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
pub struct ModularAddSubV2CoreChip<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub air: ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_SIZE>,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularAddSubV2CoreChip<NUM_LIMBS, LIMB_SIZE> {
    pub fn new(
        modulus: BigUint,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let air = ModularAddSubV2CoreAir::new(
            modulus,
            range_checker.bus().index,
            range_checker.range_max_bits(),
            offset,
        );
        Self { air, range_checker }
    }
}

pub struct ModularAddSubV2CoreRecord {
    pub x: BigUint,
    pub y: BigUint,
    pub is_add_flag: bool,
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize, I> VmCoreChip<F, I>
    for ModularAddSubV2CoreChip<NUM_LIMBS, LIMB_SIZE>
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<[[[F; NUM_LIMBS]; 1]; 2]>,
    I::Writes: From<[[F; NUM_LIMBS]; 1]>,
{
    type Record = ModularAddSubV2CoreRecord;
    type Air = ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_SIZE>;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_index = opcode - self.air.offset;
        let data: [[[F; NUM_LIMBS]; 1]; 2] = reads.into();
        let x = data[0][0].map(|x| x.as_canonical_u32());
        let y = data[1][0].map(|x| x.as_canonical_u32());

        let x_biguint = limbs_to_biguint(&x, LIMB_SIZE);
        let y_biguint = limbs_to_biguint(&y, LIMB_SIZE);

        let opcode = ModularArithmeticOpcode::from_usize(local_opcode_index);
        let is_add_flag = match opcode {
            ModularArithmeticOpcode::ADD => true,
            ModularArithmeticOpcode::SUB => false,
            _ => panic!("Unsupported opcode: {:?}", opcode),
        };

        let vars = self.air.expr.execute(
            vec![x_biguint.clone(), y_biguint.clone()],
            vec![is_add_flag],
        );
        assert_eq!(vars.len(), 1);
        let z_biguint = vars[0].clone();
        let z_limbs = biguint_to_limbs::<NUM_LIMBS>(z_biguint, LIMB_SIZE);

        Ok((
            AdapterRuntimeContext {
                to_pc: None,
                writes: [z_limbs.map(|x| F::from_canonical_u32(x))].into(),
            },
            ModularAddSubV2CoreRecord {
                x: x_biguint,
                y: y_biguint,
                is_add_flag,
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "ModularAddSub".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let input = (
            vec![record.x, record.y],
            self.range_checker.clone(),
            vec![record.is_add_flag],
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
