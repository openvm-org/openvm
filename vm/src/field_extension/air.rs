use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use itertools::izip;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::FieldExtensionArithmeticCols;
use crate::{
    arch::{
        bridge::ExecutionBridge,
        instructions::Opcode::{BBE4DIV, BBE4MUL, FE4ADD, FE4SUB},
    },
    field_extension::chip::FieldExtensionArithmetic,
    memory::offline_checker::MemoryBridge,
};

/// Field extension arithmetic chip.
///
/// Handles arithmetic opcodes over the extension field defined by the irreducible polynomial x^4 - 11.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct FieldExtensionArithmeticAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl AirConfig for FieldExtensionArithmeticAir {
    type Cols<T> = FieldExtensionArithmeticCols<T>;
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldExtensionArithmeticAir {}
impl<F: Field> BaseAir<F> for FieldExtensionArithmeticAir {
    fn width(&self) -> usize {
        FieldExtensionArithmeticCols::<F>::get_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for FieldExtensionArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local_cols = FieldExtensionArithmeticCols::from_iter(&mut local.iter().copied());

        let FieldExtensionArithmeticCols { io, aux } = local_cols;

        let flags = [aux.is_add, aux.is_sub, aux.is_mul, aux.is_div];
        let opcodes = [FE4ADD, FE4SUB, BBE4MUL, BBE4DIV];
        let results = [
            FieldExtensionArithmetic::add(io.x, io.y),
            FieldExtensionArithmetic::subtract(io.x, io.y),
            FieldExtensionArithmetic::multiply(io.x, io.y),
            FieldExtensionArithmetic::multiply(io.x, aux.divisor_inv),
        ];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results[:,j]` equals `io.z[j]` for each `j`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the inverse of `io.y`.

        let mut flag_sum = AB::Expr::zero();
        let mut expected_opcode = AB::Expr::zero();
        let mut expected_result = [
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
        ];
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            flag_sum += flag.into();
            expected_opcode += flag * AB::F::from_canonical_u32(opcode as u32);

            for (j, result_part) in result.into_iter().enumerate() {
                expected_result[j] += flag * result_part;
            }
        }
        builder.assert_eq(flag_sum, aux.is_valid);
        for (z_j, expected_result_j) in izip!(io.z, expected_result) {
            builder.assert_eq(z_j, expected_result_j);
        }

        builder.assert_bool(aux.is_valid);

        // constrain aux.divisor_inv: y * y^(-1) = 1
        let y_times_y_inv = FieldExtensionArithmetic::multiply(io.y, aux.divisor_inv);
        for (i, prod_i) in y_times_y_inv.into_iter().enumerate() {
            if i == 0 {
                builder.assert_eq(aux.is_div, prod_i);
            } else {
                builder.assert_zero(prod_i);
            }
        }

        let local_cols = FieldExtensionArithmeticCols { aux, io };
        self.eval_interactions(builder, local_cols, expected_opcode);
    }
}
