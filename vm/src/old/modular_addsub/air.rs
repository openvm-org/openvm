use std::borrow::Borrow;

use afs_primitives::bigint::{
    check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
    OverflowInt,
};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::ModularAddSubCols;
use crate::{
    arch::{instructions::ModularArithmeticOpcode, ExecutionBridge},
    system::memory::offline_checker::MemoryBridge,
};

#[derive(Debug, Clone)]
pub struct ModularAddSubAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub(super) subair: CheckCarryModToZeroSubAir,

    pub(super) offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> PartitionedBaseAir<F>
    for ModularAddSubAir<NUM_LIMBS, LIMB_BITS>
{
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ModularAddSubAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ModularAddSubCols::<F, NUM_LIMBS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ModularAddSubAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for ModularAddSubAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let ModularAddSubCols::<AB::Var, NUM_LIMBS> { io, aux } = (*local).borrow();

        // we assume aux.is_sub is represented aux.is_valid - aux.is_add
        builder.assert_bool(aux.is_add);
        builder.assert_bool(aux.is_valid - aux.is_add);
        let expected_opcode = AB::Expr::from_canonical_u8(ModularArithmeticOpcode::SUB as u8)
            + aux.is_add
                * (AB::Expr::from_canonical_u8(ModularArithmeticOpcode::ADD as u8)
                    - AB::Expr::from_canonical_u8(ModularArithmeticOpcode::SUB as u8));

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.x.data.data.to_vec(),
            LIMB_BITS,
        );
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.y.data.data.to_vec(),
            LIMB_BITS,
        );
        let z_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.z.data.data.to_vec(),
            LIMB_BITS,
        );

        let y_cond_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(
            io.y.data
                .data
                .map(|y| AB::Expr::two() * y * aux.is_add)
                .to_vec(),
            LIMB_BITS,
        );

        // for addition we get y_overflow = y_cond_overflow - y_overflow
        // for subtraction we get -y_overflow = y_cond_overflow - y_overflow
        // Thus, the value of expr will be correct
        let expr = x_overflow - y_overflow + y_cond_overflow - z_overflow;

        self.subair.constrain_carry_mod_to_zero(
            builder,
            expr,
            CheckCarryModToZeroCols {
                carries: aux.carries.to_vec(),
                quotient: vec![aux.q],
            },
            aux.is_valid,
        );

        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
