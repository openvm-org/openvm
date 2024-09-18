use std::borrow::Borrow;

use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::interaction::InteractionBuilder;
use derive_new::new;
use itertools::izip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use poseidon2_air::poseidon2::Poseidon2Air;

use super::{columns::Poseidon2VmCols, CHUNK, WIDTH};
use crate::{
    arch::bus::ExecutionBus,
    memory::{offline_checker::MemoryBridge, MemoryAddress},
};

/// Poseidon2 Air, VM version.
///
/// Carries the subair for subtrace generation. Sticking to the conventions, this struct carries no state.
/// `direct` determines whether direct interactions are enabled. By default they are on.
#[derive(Clone, new, Debug)]
pub struct Poseidon2VmAir<T> {
    pub inner: Poseidon2Air<WIDTH, T>,
    pub execution_bus: ExecutionBus,
    pub memory_bridge: MemoryBridge,
    pub direct: bool, // Whether direct interactions are enabled.
}

impl<F> AirConfig for Poseidon2VmAir<F> {
    type Cols<T> = Poseidon2VmCols<T>;
}

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

        // boolean constraints for alloc/cmp markers
        // these constraints hold for current trace generation mechanism but are in actuality not necessary
        builder.assert_bool(cols.io.is_opcode);
        builder.assert_bool(cols.io.is_direct);
        builder.assert_bool(cols.io.cmp);
        // can only be comparing if row is allocated
        builder.assert_eq(cols.io.is_opcode * cols.io.cmp, cols.io.cmp);
        // if io.cmp is false, then constrain rhs = lhs + CHUNK
        builder.when(cols.io.is_opcode - cols.io.cmp).assert_eq(
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
            [cols.io.is_opcode, cols.io.is_opcode, cols.io.cmp],
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
            .eval(builder, cols.io.is_opcode - cols.io.cmp);

        self.eval_interactions(
            builder,
            cols.io,
            internal_io,
            AB::Expr::from_canonical_usize(timestamp_delta),
        );
    }
}
