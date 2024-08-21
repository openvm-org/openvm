use afs_primitives::{
    sub_chip::AirConfig,
    utils::{and, not},
};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::FieldExtensionArithmeticCols;
use crate::{
    cpu::OpCode::{BBE4INV, BBE4MUL, FE4ADD, FE4SUB},
    field_extension::chip::FieldExtensionArithmetic,
    memory::offline_checker::bridge::MemoryOfflineChecker,
};

/// Field extension arithmetic chip.
///
/// Handles arithmetic opcodes over the extension field defined by the irreducible polynomial x^4 - 11.
#[derive(Clone)]
pub struct FieldExtensionArithmeticAir<const WORD_SIZE: usize> {
    pub(crate) mem_oc: MemoryOfflineChecker,
}

impl<const WORD_SIZE: usize> AirConfig for FieldExtensionArithmeticAir<WORD_SIZE> {
    type Cols<T> = FieldExtensionArithmeticCols<WORD_SIZE, T>;
}

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for FieldExtensionArithmeticAir<WORD_SIZE> {
    fn width(&self) -> usize {
        FieldExtensionArithmeticCols::<WORD_SIZE, F>::get_width(self)
    }
}

impl<const WORD_SIZE: usize, AB: InteractionBuilder> Air<AB>
    for FieldExtensionArithmeticAir<WORD_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local_cols = FieldExtensionArithmeticCols::<WORD_SIZE, AB::Var>::from_iter(
            &mut local.iter().copied(),
            &self.mem_oc.timestamp_lt_air,
        );

        let FieldExtensionArithmeticCols { io, aux } = local_cols;

        // TODO[zach]: Support DIV directly instead of INV.

        let flags = [aux.is_add, aux.is_sub, aux.is_mul, aux.is_inv];
        let opcodes = [FE4ADD, FE4SUB, BBE4MUL, BBE4INV];
        let results = [
            FieldExtensionArithmetic::add(io.x, io.y),
            FieldExtensionArithmetic::subtract(io.x, io.y),
            FieldExtensionArithmetic::multiply(io.x, io.y),
            aux.inv.map(Into::into),
        ];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results[:,j]` equals `io.z[j]` for each `j`.

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
        builder.assert_one(flag_sum);
        builder.assert_eq(io.opcode, expected_opcode);
        for (z_j, expected_result_j) in izip!(io.z, expected_result) {
            builder.assert_eq(z_j, expected_result_j);
        }

        builder.assert_bool(aux.is_valid);
        // valid_y_read is 1 iff is_valid and not is_inv
        // the previous constraint along with this one imply valid_y_read is boolean
        builder.assert_eq(
            aux.valid_y_read,
            and(aux.is_valid.into(), not(aux.is_inv.into())),
        );

        // constrain inverse using multiplication: x * x^(-1) = 1
        // ignores when not inv compute (will fail if x = 0 and try to compute inv)
        let x_times_x_inv = FieldExtensionArithmetic::multiply(io.x, aux.inv);
        for (i, prod_i) in x_times_x_inv.into_iter().enumerate() {
            if i == 0 {
                builder.when(aux.is_inv).assert_one(prod_i);
            } else {
                builder.assert_zero(prod_i);
            }
        }

        let local_cols = FieldExtensionArithmeticCols { aux, io };
        self.eval_interactions(builder, local_cols);
    }
}
