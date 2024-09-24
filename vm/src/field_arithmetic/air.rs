use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::FieldArithmeticCols;
use crate::{
    arch::{
        bridge::ExecutionBridge,
        instructions::{Opcode, OpcodeEncoder, FIELD_ARITHMETIC_INSTRUCTIONS},
    },
    memory::offline_checker::MemoryBridge,
};

#[derive(Clone, Debug)]
pub struct FieldArithmeticAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub(super) opcode_encoder: OpcodeEncoder<2>,
}

impl FieldArithmeticAir {
    pub const TIMESTAMP_DELTA: usize = 3;
}

impl AirConfig for FieldArithmeticAir {
    type Cols<T> = FieldArithmeticCols<T>;
}

impl<F: Field> BaseAir<F> for FieldArithmeticAir {
    fn width(&self) -> usize {
        FieldArithmeticCols::<F>::get_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for FieldArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = FieldArithmeticCols::from_iter(&mut local.iter().copied());

        let FieldArithmeticCols { io, aux } = local;

        let x = io.x.value;
        let y = io.y.value;
        let z = io.z.value;

        let flags = aux.flags;
        let results = [x + y, x - y, x * y, x * aux.divisor_inv];

        let encoder = self.opcode_encoder.initialize(builder, flags);

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results` equals `io.z`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the multiplicative inverse of `io.y`.

        let mut expected_opcode = AB::Expr::zero();
        let mut expected_result = AB::Expr::zero();
        for (opcode, result) in izip!(FIELD_ARITHMETIC_INSTRUCTIONS, results) {
            let flag = encoder.expression_for(opcode);

            expected_opcode += flag.clone() * AB::Expr::from_canonical_u32(opcode as u32);
            expected_result += flag * result;
        }
        builder.assert_eq(z, expected_result);
        builder.assert_bool(aux.is_valid);

        builder.assert_eq(encoder.expression_for(Opcode::FDIV), y * aux.divisor_inv);

        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
