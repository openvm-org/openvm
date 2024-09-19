use std::borrow::Borrow;

use afs_primitives::bigint::{
    check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
    utils::{big_uint_to_limbs, secp256k1_coord_prime},
    OverflowInt,
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::ModularArithmeticCols;
use crate::{
    arch::{bus::ExecutionBus, instructions::Opcode},
    memory::offline_checker::MemoryBridge,
};

#[derive(Debug, Clone)]
pub struct ModularArithmeticAir<const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub(super) execution_bus: ExecutionBus,
    pub(super) memory_bridge: MemoryBridge,
    pub(super) subair: CheckCarryModToZeroSubAir,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_SIZE: usize> BaseAir<F>
    for ModularArithmeticAir<NUM_LIMBS, LIMB_SIZE>
{
    fn width(&self) -> usize {
        ModularArithmeticCols::<F, NUM_LIMBS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_SIZE: usize> Air<AB>
    for ModularArithmeticAir<NUM_LIMBS, LIMB_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let ModularArithmeticCols::<AB::Var, NUM_LIMBS> { io, aux } = (*local).borrow();

        // we assume aux.is_sub is represented aaux.is_valid - aux.is_add
        builder.assert_bool(aux.is_add);
        builder.assert_bool(aux.is_valid - aux.is_add);
        let expected_opcode = if self.subair.modulus_limbs
            == big_uint_to_limbs(&secp256k1_coord_prime(), LIMB_SIZE)
        {
            AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_SUB as u8)
                + aux.is_add
                    * (AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_ADD as u8)
                        - AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_SUB as u8))
        } else {
            AB::Expr::from_canonical_u8(Opcode::SECP256K1_SCALAR_SUB as u8)
                + aux.is_add
                    * (AB::Expr::from_canonical_u8(Opcode::SECP256K1_SCALAR_ADD as u8)
                        - AB::Expr::from_canonical_u8(Opcode::SECP256K1_SCALAR_SUB as u8))
        };

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.x.data.data.to_vec(),
            LIMB_SIZE,
        );
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.y.data.data.to_vec(),
            LIMB_SIZE,
        );
        let r_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.z.data.data.to_vec(),
            LIMB_SIZE,
        );

        let y_cond_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(
            io.y.data
                .data
                .map(|y| AB::Expr::two() * y * aux.is_add)
                .to_vec(),
            LIMB_SIZE,
        );

        // for addition we get y_overflow = y_cond_overflow - y_overflow
        // for subtraction we get -y_overflow = y_cond_overflow - y_overflow
        // Thus, the value of expr will be correct
        let expr = x_overflow - y_overflow + y_cond_overflow - r_overflow;

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
