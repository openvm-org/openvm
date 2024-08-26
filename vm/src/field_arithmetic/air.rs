use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::FieldArithmeticCols;
use crate::{
    arch::{
        bridge::ExecutionBus,
        instructions::Opcode::{FADD, FDIV, FMUL, FSUB},
    },
    memory::offline_checker::bridge::MemoryOfflineChecker,
};

#[derive(Clone, Copy, Debug)]
pub struct FieldArithmeticAir {
    pub(super) execution_bus: ExecutionBus,
    pub(super) mem_oc: MemoryOfflineChecker,
}

impl FieldArithmeticAir {
    pub const TIMESTAMP_DELTA: usize = 3;
}

impl AirConfig for FieldArithmeticAir {
    type Cols<T> = FieldArithmeticCols<T>;
}

impl<F: Field> BaseAir<F> for FieldArithmeticAir {
    fn width(&self) -> usize {
        FieldArithmeticCols::<F>::get_width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for FieldArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = FieldArithmeticCols::from_iter(&mut local.iter().copied(), self);

        let FieldArithmeticCols { io, aux } = local;

        let x = io.operand1.value;
        let y = io.operand2.value;
        let z = io.result.value;

        let flags = [aux.is_add, aux.is_sub, aux.is_mul, aux.is_div];
        let opcodes = [FADD, FSUB, FMUL, FDIV];
        let results = [x + y, x - y, x * y, x * aux.divisor_inv];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results` equals `io.z`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the multiplicative inverse of `io.y`.

        let mut flag_sum = AB::Expr::zero();
        let mut expected_opcode = AB::Expr::zero();
        let mut expected_result = AB::Expr::zero();
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            flag_sum += flag.into();
            expected_opcode += flag * AB::Expr::from_canonical_u32(opcode as u32);
            expected_result += flag * result;
        }
        builder.assert_one(flag_sum);
        builder.assert_eq(io.opcode, expected_opcode);
        builder.assert_eq(z, expected_result);

        builder.assert_eq(aux.is_div, y * aux.divisor_inv);

        self.eval_interactions(builder, io, aux);
    }
}
