use std::{array::from_fn, borrow::Borrow};

use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::interaction::InteractionBuilder;
use derive_new::new;
use itertools::izip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use poseidon2_air::poseidon2::Poseidon2Air;

use super::{columns::Poseidon2VmCols, CHUNK};
use crate::{
    cpu::OpCode,
    memory::{
        manager::{access_cell::AccessCell, operation::MemoryOperation},
        offline_checker::bridge::NewMemoryOfflineChecker,
    },
};

/// Poseidon2 Air, VM version.
///
/// Carries the subair for subtrace generation. Sticking to the conventions, this struct carries no state.
/// `direct` determines whether direct interactions are enabled. By default they are on.
#[derive(new)]
pub struct Poseidon2VmAir<const WIDTH: usize, const WORD_SIZE: usize, F: Clone> {
    pub inner: Poseidon2Air<WIDTH, F>,
    pub mem_oc: NewMemoryOfflineChecker,
    pub direct: bool, // Whether direct interactions are enabled.
}

impl<const WIDTH: usize, const WORD_SIZE: usize, F: Clone> AirConfig
    for Poseidon2VmAir<WIDTH, WORD_SIZE, F>
{
    type Cols<T> = Poseidon2VmCols<WIDTH, WORD_SIZE, T>;
}

impl<const WIDTH: usize, const WORD_SIZE: usize, F: Field> BaseAir<F>
    for Poseidon2VmAir<WIDTH, WORD_SIZE, F>
{
    fn width(&self) -> usize {
        Poseidon2VmCols::<WIDTH, WORD_SIZE, F>::width(self)
    }
}

impl<AB: InteractionBuilder, const WIDTH: usize, const WORD_SIZE: usize> Air<AB>
    for Poseidon2VmAir<WIDTH, WORD_SIZE, AB::F>
{
    /// Checks and constrains multiplicity indicators, and does subair evaluation
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[<AB>::Var] = (*local).borrow();

        let mut cols = Poseidon2VmCols::<WIDTH, WORD_SIZE, AB::Var>::from_slice(local, self);

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
            cols.aux.rhs,
            cols.aux.lhs + AB::F::from_canonical_usize(CHUNK),
        );

        // Memory access constraints
        let chunks: usize = WIDTH / 2;

        let mut clk_offset = 0;
        // read addresses when is_opcode:
        // dst <- [a]_d, lhs <- [b]_d
        // Only when opcode is COMPRESS is rhs <- [c]_d read
        for (io_addr, aux_addr, count) in izip!(
            [cols.io.a, cols.io.b, cols.io.c],
            [cols.aux.dst, cols.aux.lhs, cols.aux.rhs],
            [cols.io.is_opcode, cols.io.is_opcode, cols.io.cmp,]
        ) {
            let clk = cols.io.clk + AB::F::from_canonical_usize(clk_offset);
            clk_offset += 1;

            let mem_op = MemoryOperation::<WORD_SIZE, AB::Expr>::new(
                cols.io.d.into(),
                io_addr.into(),
                AB::Expr::zero(),
                AccessCell::new(from_fn(|_| aux_addr.into()), clk),
                count.into(),
            );

            self.mem_oc
                .subair_eval(builder, mem_op, cols.aux.mem_oc_aux_cols.remove(0));
        }

        // READ
        for i in 0..WIDTH {
            let clk = cols.io.clk + AB::F::from_canonical_usize(clk_offset);
            clk_offset += 1;

            let pointer = if i < chunks {
                cols.aux.lhs
            } else {
                cols.aux.rhs
            } + AB::F::from_canonical_usize(if i < chunks { i } else { i - chunks });

            let mem_op = MemoryOperation::<WORD_SIZE, AB::Expr>::new(
                cols.io.e.into(),
                pointer,
                AB::Expr::zero(),
                AccessCell::new(from_fn(|_| cols.aux.internal.io.input[i].into()), clk),
                cols.io.is_opcode.into(),
            );

            self.mem_oc
                .subair_eval(builder, mem_op, cols.aux.mem_oc_aux_cols.remove(0));
        }

        // WRITE
        for i in 0..WIDTH {
            let clk = cols.io.clk + AB::F::from_canonical_usize(clk_offset);
            clk_offset += 1;

            let pointer = cols.aux.dst + AB::F::from_canonical_usize(i);

            let count = if i < chunks {
                cols.io.is_opcode.into()
            } else {
                cols.io.is_opcode - cols.io.cmp
            };

            let mem_op = MemoryOperation::<WORD_SIZE, AB::Expr>::new(
                cols.io.e.into(),
                pointer,
                AB::Expr::one(),
                AccessCell::new(from_fn(|_| cols.aux.internal.io.output[i].into()), clk),
                count,
            );

            self.mem_oc
                .subair_eval(builder, mem_op, cols.aux.mem_oc_aux_cols.remove(0));
        }
    }
}
