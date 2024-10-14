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
        AdapterRuntimeContext, Result, Rv32HeapAdapter, Rv32HeapAdapterCols,
        Rv32HeapAdapterInterface, VmAdapterInterface, VmCoreChip,
    },
    program::Instruction,
    utils::{biguint_to_limbs, limbs_to_biguint},
};

#[derive(Clone)]
pub struct ModularAddSubV2CoreChip<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub chip: FieldExprChip,

    pub offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularAddSubV2CoreChip<NUM_LIMBS, LIMB_SIZE> {
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
        let x3 = x1.clone() + x2.clone();
        let x4 = x1 - x2;
        let is_add_flag = builder.borrow_mut().new_flag();
        let mut x5 = FieldVariable::select(is_add_flag, &x3, &x4);
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
    VmCoreChip<F, Rv32HeapAdapter<F, NUM_LIMBS, NUM_LIMBS>>
    for ModularAddSubV2CoreChip<NUM_LIMBS, LIMB_SIZE>
{
    type Record = ();
    type Air = FieldExprChip;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(
        AdapterRuntimeContext<F, Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS>>,
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
        let is_add_flag = match opcode {
            ModularArithmeticOpcode::ADD => true,
            ModularArithmeticOpcode::SUB => false,
            _ => panic!("Unsupported opcode: {:?}", opcode),
        };

        let vars = self
            .chip
            .execute(vec![x_biguint, y_biguint], vec![is_add_flag]);
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
        &self.chip
    }
}
