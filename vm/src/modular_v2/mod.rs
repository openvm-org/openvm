use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExprChip, FieldVariableConfig};
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;

use crate::{
    arch::{AdapterRuntimeContext, Rv32HeapAdapter, Rv32HeapAdapterInterface, VmCore},
    utils::{biguint_to_limbs, limbs_to_biguint},
};

#[cfg(test)]
mod tests;

pub const FIELD_ELEMENT_BITS: usize = 30;
const LIMB_BITS: usize = 8;

#[derive(Clone)]
pub struct ModularConfig<const NUM_LIMBS: usize> {}
impl<const NUM_LIMBS: usize> FieldVariableConfig for ModularConfig<NUM_LIMBS> {
    fn canonical_limb_bits() -> usize {
        LIMB_BITS
    }

    fn max_limb_bits() -> usize {
        FIELD_ELEMENT_BITS - 1
    }

    fn num_limbs_per_field_element() -> usize {
        NUM_LIMBS
    }
}

#[derive(Clone)]
pub struct ModularAddSubV2Chip<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub chip: FieldExprChip,
    modulus: BigUint,
}

impl<const NUM_LIMBS: usize, const LIMB_SIZE: usize> ModularAddSubV2Chip<NUM_LIMBS, LIMB_SIZE> {
    pub fn new(modulus: BigUint, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
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
            modulus.clone(),
            LIMB_SIZE,
            NUM_LIMBS,
            range_checker.range_max_bits(),
        );
        let builder = Rc::new(RefCell::new(builder));
        let x1 = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let x2 = ExprBuilder::new_input::<ModularConfig<NUM_LIMBS>>(builder.clone());
        let mut x3 = x1 + x2;
        x3.save();
        let builder = builder.borrow().clone();

        let chip = FieldExprChip {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker,
        };
        Self { chip, modulus }
    }
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    VmCore<F, Rv32HeapAdapter<F, NUM_LIMBS, NUM_LIMBS>>
    for ModularAddSubV2Chip<NUM_LIMBS, LIMB_SIZE>
{
    type Record = ();
    type Air = FieldExprChip;

    fn execute_instruction(
        &self,
        _instruction: &crate::program::Instruction<F>,
        _from_pc: F,
        reads: <Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS> as crate::arch::VmAdapterInterface<F>>::Reads,
    ) -> crate::arch::Result<(
        AdapterRuntimeContext<F, Rv32HeapAdapterInterface<F, NUM_LIMBS, NUM_LIMBS>>,
        Self::Record,
    )> {
        let (x, y) = reads;
        let x = x.map(|x| x.as_canonical_u32());
        let y = y.map(|x| x.as_canonical_u32());

        let x_biguint = limbs_to_biguint(&x, LIMB_SIZE);
        let y_biguint = limbs_to_biguint(&y, LIMB_SIZE);

        // TODO: chip (expr builder) should be able to handle this.
        let z_biguint = (x_biguint + y_biguint) % &self.modulus;
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
