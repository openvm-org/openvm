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
    memory::{
        offline_checker::bridge::{MemoryBridge, MemoryOfflineChecker},
        MemoryAddress,
    },
};

/// Poseidon2 Air, VM version.
///
/// Carries the subair for subtrace generation. Sticking to the conventions, this struct carries no state.
/// `direct` determines whether direct interactions are enabled. By default they are on.
#[derive(Clone, new, Debug)]
pub struct Poseidon2VmAir<T> {
    pub inner: Poseidon2Air<WIDTH, T>,
    pub execution_bus: ExecutionBus,
    pub mem_oc: MemoryOfflineChecker,
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

        self.eval_interactions(builder, cols.io, &cols.aux);
        self.inner
            .eval_without_interactions(builder, cols.aux.internal.io, cols.aux.internal.aux);

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
        let memory_bridge = MemoryBridge::new(self.mem_oc);
        let mut clk_offset = 0;
        let timestamp_base = cols.io.timestamp;

        // read addresses when is_opcode:
        // dst <- [a]_d, lhs <- [b]_d
        // Only when opcode is COMPRESS is rhs <- [c]_d read
        for (io_addr, aux_addr, count, mem_aux) in izip!(
            [cols.io.a, cols.io.b, cols.io.c],
            [cols.aux.dst_ptr, cols.aux.lhs_ptr, cols.aux.rhs_ptr],
            [cols.io.is_opcode, cols.io.is_opcode, cols.io.cmp],
            cols.aux.ptr_aux_cols,
        ) {
            let clk = timestamp_base + AB::F::from_canonical_usize(clk_offset);
            clk_offset += 1;

            memory_bridge
                .read(
                    MemoryAddress::new(cols.io.d, io_addr),
                    [aux_addr],
                    clk,
                    mem_aux.clone(),
                )
                .eval(builder, count);
        }

        // First input chunk.
        {
            let clk = cols.io.timestamp + AB::F::from_canonical_usize(clk_offset);
            clk_offset += CHUNK;

            memory_bridge
                .read(
                    MemoryAddress::new(cols.io.e, cols.aux.lhs_ptr),
                    cols.aux.internal.io.input[..CHUNK].try_into().unwrap(),
                    clk,
                    cols.aux.input_aux_cols[0].clone(),
                )
                .eval(builder, cols.io.is_opcode);
        }
        // Second input chunk.
        {
            let clk = cols.io.timestamp + AB::F::from_canonical_usize(clk_offset);
            clk_offset += CHUNK;

            memory_bridge
                .read(
                    MemoryAddress::new(cols.io.e, cols.aux.rhs_ptr),
                    cols.aux.internal.io.input[CHUNK..].try_into().unwrap(),
                    clk,
                    cols.aux.input_aux_cols[1].clone(),
                )
                .eval(builder, cols.io.is_opcode);
        }
        // First output chunk.
        {
            let clk = cols.io.timestamp + AB::F::from_canonical_usize(clk_offset);
            clk_offset += CHUNK;

            memory_bridge
                .write(
                    MemoryAddress::new(cols.io.e, cols.aux.dst_ptr),
                    cols.aux.internal.io.output[..CHUNK].try_into().unwrap(),
                    clk,
                    cols.aux.output_aux_cols[0].clone(),
                )
                .eval(builder, cols.io.is_opcode);
        }
        // Second output chunk.
        {
            let clk = cols.io.timestamp + AB::F::from_canonical_usize(clk_offset);

            let pointer = cols.aux.dst_ptr + AB::F::from_canonical_usize(CHUNK);
            memory_bridge
                .write(
                    MemoryAddress::new(cols.io.e, pointer),
                    cols.aux.internal.io.output[CHUNK..].try_into().unwrap(),
                    clk,
                    cols.aux.output_aux_cols[1].clone(),
                )
                .eval(builder, cols.io.is_opcode - cols.io.cmp);
        }
    }
}
