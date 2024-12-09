use std::borrow::Borrow;

use ax_poseidon2_air::poseidon2::Poseidon2Air;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{AbstractField, Field},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use derive_new::new;
use itertools::izip;

use super::{columns::Poseidon2VmCols, CHUNK, WIDTH};
use crate::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, MemoryAddress},
};

/// Poseidon2 Air, VM version.
///
/// Carries the subair for subtrace generation. Sticking to the conventions, this struct carries no state.
/// `direct` determines whether direct interactions are enabled. By default they are on.
#[derive(Clone, new, Debug)]
pub struct Poseidon2VmAir<T> {
    pub inner: Poseidon2Air<WIDTH, T>,
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub direct_bus: Option<usize>, // Whether direct interactions are enabled.

    pub(super) offset: usize,
}

impl<F: Field> BaseAirWithPublicValues<F> for Poseidon2VmAir<F> {}
impl<F: Field> PartitionedBaseAir<F> for Poseidon2VmAir<F> {}
impl<F: Field> BaseAir<F> for Poseidon2VmAir<F> {
    fn width(&self) -> usize {
        Poseidon2VmCols::<F>::width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for Poseidon2VmAir<AB::F> {
    /// Checks and constrains multiplicity indicators, and does subair evaluation
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[<AB>::Var] = (*local).borrow();

        let cols = Poseidon2VmCols::<AB::Var>::from_slice(local, self);
        let internal_io = cols.aux.internal.io;

        self.inner.eval_without_interactions(
            builder,
            cols.aux.internal.io,
            cols.aux.internal.aux.into_expr::<AB>(),
        );

        builder.assert_bool(cols.io.is_opcode);
        builder.assert_bool(cols.io.is_compress_opcode);
        builder.assert_bool(cols.io.is_compress_direct);

        // Both opcode and compress_direct cannot be true
        builder.assert_zero(cols.io.is_opcode * cols.io.is_compress_direct);

        builder
            .when(cols.io.is_compress_opcode)
            .assert_one(cols.io.is_opcode);
        let is_permute_opcode = cols.io.is_opcode - cols.io.is_compress_opcode;

        // if permute instruction, the rhs_ptr should be contiguous with lhs_ptr
        builder.when(is_permute_opcode.clone()).assert_eq(
            cols.aux.rhs_ptr,
            cols.aux.lhs_ptr + AB::F::from_canonical_usize(CHUNK),
        );

        // Memory access constraints
        let timestamp = cols.io.timestamp;
        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // read addresses when is_opcode:
        // dst <- [a]_d, lhs <- [b]_d
        // Only when opcode is COMPRESS is rhs <- [c]_d read
        for (io_addr, aux_addr, count, mem_aux) in izip!(
            [cols.io.a, cols.io.b, cols.io.c],
            [cols.aux.dst_ptr, cols.aux.lhs_ptr, cols.aux.rhs_ptr],
            [
                cols.io.is_opcode,
                cols.io.is_opcode,
                cols.io.is_compress_opcode
            ],
            &cols.aux.ptr_aux_cols,
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(cols.io.d, io_addr),
                    [aux_addr],
                    timestamp_pp(),
                    mem_aux,
                )
                .eval(builder, count);
        }

        let [input1_aux_cols, input2_aux_cols] = cols.aux.input_aux_cols;
        let [output1_aux_cols, output2_aux_cols] = cols.aux.output_aux_cols;

        // First input chunk.
        self.memory_bridge
            .read(
                MemoryAddress::new(cols.io.e, cols.aux.lhs_ptr),
                cols.aux.internal.io.input[..CHUNK].try_into().unwrap(),
                timestamp_pp(),
                &input1_aux_cols,
            )
            .eval(builder, cols.io.is_opcode);

        // Second input chunk.
        self.memory_bridge
            .read(
                MemoryAddress::new(cols.io.e, cols.aux.rhs_ptr),
                cols.aux.internal.io.input[CHUNK..].try_into().unwrap(),
                timestamp_pp(),
                &input2_aux_cols,
            )
            .eval(builder, cols.io.is_opcode);

        // First output chunk.
        self.memory_bridge
            .write(
                MemoryAddress::new(cols.io.e, cols.aux.dst_ptr),
                cols.aux.internal.io.output[..CHUNK].try_into().unwrap(),
                timestamp_pp(),
                &output1_aux_cols,
            )
            .eval(builder, cols.io.is_opcode);

        // Second output chunk.
        let pointer = cols.aux.dst_ptr + AB::F::from_canonical_usize(CHUNK);
        self.memory_bridge
            .write(
                MemoryAddress::new(cols.io.e, pointer),
                cols.aux.internal.io.output[CHUNK..].try_into().unwrap(),
                timestamp_pp(),
                &output2_aux_cols,
            )
            .eval(builder, is_permute_opcode);

        self.eval_interactions(
            builder,
            cols.io,
            internal_io,
            AB::Expr::from_canonical_usize(timestamp_delta),
        );
    }
}
