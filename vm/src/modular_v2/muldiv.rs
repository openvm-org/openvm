use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExprChip, FieldVariable};
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;

use super::{ModularConfig, FIELD_ELEMENT_BITS};
use crate::{
    arch::{
        instructions::{ModularArithmeticOpcode, UsizeOpcode},
        InstructionOutput, MachineAdapterInterface, MachineIntegration, Result, Rv32HeapAdapter,
        Rv32HeapAdapterCols, Rv32HeapAdapterInterface,
    },
    program::Instruction,
    utils::{biguint_to_limbs, limbs_to_biguint},
};

#[derive(Clone)]
pub struct ModularMulDivV2Chip<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub chip: FieldExprChip,

    pub offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularMulDivV2Chip<NUM_LIMBS, LIMB_SIZE> {
    pub fn new(
        modulus: BigUint,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        // TODO: assert modulus and NUM_LIMBS are consistent with each other
        let bus = range_checker.bus();
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            LIMB_SIZE,
            bus.index,
            bus.range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        let builder = ExprBuilder::new(
            modulus,
            LIMB_SIZE,
            NUM_LIMBS,
            range_checker.range_max_bits(),
        );
        let builder = Rc::new(RefCell::new(builder));
        let x1 = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let x2 = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        // constraint is x * x2 = z
        // where x = if is_mul_flag then x1 else x3
        // and z = if is_mul_flag then x3 else x1
        let mut x3 = x1.clone() * x2.clone();
        x3.save(); // div auto save, so this has to save to match.
        let x4 = x1 / x2;
        let is_mul_flag = builder.borrow_mut().new_flag();
        let mut x5 = FieldVariable::select(is_mul_flag, &x3, &x4);
        x5.save();
        let builder = builder.borrow().clone();

        let chip = FieldExprChip {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker,
        };
        Self { chip, offset }
    }
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    MachineIntegration<F, Rv32HeapAdapter<F, NUM_LIMBS, NUM_LIMBS>>
    for ModularMulDivV2Chip<NUM_LIMBS, LIMB_SIZE>
{
    type Record = ();
    type Cols<T> = Vec<T>;
    type Air = FieldExprChip;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS> as MachineAdapterInterface<F>>::Reads,
    ) -> Result<(
        InstructionOutput<F, Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS>>,
        Self::Record,
    )> {
        let Instruction { opcode, .. } = instruction.clone();
        let opcode = opcode - self.offset;
        let (x, y) = reads;
        let x = x.map(|x| x.as_canonical_u32());
        let y = y.map(|x| x.as_canonical_u32());

        let x_biguint = limbs_to_biguint(&x, LIMB_SIZE);
        let y_biguint = limbs_to_biguint(&y, LIMB_SIZE);

        let opcode = ModularArithmeticOpcode::from_usize(opcode);
        let is_mul_flag = match opcode {
            ModularArithmeticOpcode::MUL => true,
            ModularArithmeticOpcode::DIV => false,
            _ => panic!("Unsupported opcode: {:?}", opcode),
        };

        let vars = self
            .chip
            .execute(vec![x_biguint, y_biguint], vec![is_mul_flag]);
        assert_eq!(vars.len(), 3); // TODO: might be too inefficient.
        let z_biguint = vars[2].clone();
        let z_limbs = biguint_to_limbs::<NUM_LIMBS>(z_biguint, LIMB_SIZE);

        Ok((
            InstructionOutput {
                to_pc: None,
                writes: z_limbs.map(|x| F::from_canonical_u32(x)),
            },
            (),
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "todo".to_string()
    }

    fn generate_trace_row(&self, _row_slice: &mut Self::Cols<F>, _record: Self::Record) {
        todo!()
    }

    fn eval_primitive<
        AB: afs_stark_backend::interaction::InteractionBuilder<F = F>
            + p3_air::PairBuilder
            + p3_air::AirBuilderWithPublicValues,
    >(
        _air: &Self::Air,
        _builder: &mut AB,
        _local: &Self::Cols<AB::Var>,
        _local_adapter: &Rv32HeapAdapterCols<AB::Var, NUM_LIMBS, NUM_LIMBS>,
    ) -> crate::arch::IntegrationInterface<
        AB::Expr,
        Rv32HeapAdapterInterface<AB::Expr, NUM_LIMBS, NUM_LIMBS>,
    > {
        todo!()
    }

    fn air(&self) -> Self::Air {
        self.chip.clone()
    }
}
