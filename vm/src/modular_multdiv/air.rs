use std::{borrow::Borrow, iter::zip};

use afs_primitives::bigint::{
    check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
    utils::{big_uint_to_limbs, secp256k1_coord_prime},
    OverflowInt,
};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::ModularMultDivCols;
use crate::{
    arch::{bridge::ExecutionBridge, instructions::Opcode},
    memory::offline_checker::MemoryBridge,
};

#[derive(Debug, Clone)]
pub struct ModularMultDivAir<
    const CARRY_LIMBS: usize,
    const NUM_LIMBS: usize,
    const LIMB_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub(super) subair: CheckCarryModToZeroSubAir,
}

impl<F: Field, const CARRY_LIMBS: usize, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    PartitionedBaseAir<F> for ModularMultDivAir<CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
}
impl<F: Field, const CARRY_LIMBS: usize, const NUM_LIMBS: usize, const LIMB_SIZE: usize> BaseAir<F>
    for ModularMultDivAir<CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
    fn width(&self) -> usize {
        ModularMultDivCols::<F, CARRY_LIMBS, NUM_LIMBS>::width()
    }
}

impl<F: Field, const CARRY_LIMBS: usize, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    BaseAirWithPublicValues<F> for ModularMultDivAir<CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
}

impl<
        AB: InteractionBuilder,
        const CARRY_LIMBS: usize,
        const NUM_LIMBS: usize,
        const LIMB_SIZE: usize,
    > Air<AB> for ModularMultDivAir<CARRY_LIMBS, NUM_LIMBS, LIMB_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let ModularMultDivCols::<AB::Var, CARRY_LIMBS, NUM_LIMBS> { io, aux } = (*local).borrow();

        // we assume aux.is_div is represented aux.is_valid - aux.is_mult
        builder.assert_bool(aux.is_mult);
        builder.assert_bool(aux.is_valid - aux.is_mult);
        let expected_opcode = if self.subair.modulus_limbs
            == big_uint_to_limbs(&secp256k1_coord_prime(), LIMB_SIZE)
        {
            AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_DIV as u8)
                + aux.is_mult
                    * (AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_MUL as u8)
                        - AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_DIV as u8))
        } else {
            AB::Expr::from_canonical_u8(Opcode::SECP256K1_SCALAR_DIV as u8)
                + aux.is_mult
                    * (AB::Expr::from_canonical_u8(Opcode::SECP256K1_SCALAR_MUL as u8)
                        - AB::Expr::from_canonical_u8(Opcode::SECP256K1_SCALAR_DIV as u8))
        };

        // We want expr = x * y - z if the operation is mult,
        //     and expr = y * z - x if the operation is div
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            io.y.data.data.to_vec(),
            LIMB_SIZE,
        );
        let x_or_z = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(
            zip(io.x.data.data, io.z.data.data)
                .map(|(x, z)| x * aux.is_mult + z * (aux.is_valid - aux.is_mult))
                .collect(),
            LIMB_SIZE,
        );
        let z_or_x = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Expr>(
            zip(io.x.data.data, io.z.data.data)
                .map(|(x, z)| z * aux.is_mult + x * (aux.is_valid - aux.is_mult))
                .collect(),
            LIMB_SIZE,
        );

        let expr = x_or_z * y_overflow - z_or_x;

        self.subair.constrain_carry_mod_to_zero(
            builder,
            expr,
            CheckCarryModToZeroCols {
                carries: aux.carries.to_vec(),
                quotient: aux.q.to_vec(),
            },
            aux.is_valid,
        );

        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
