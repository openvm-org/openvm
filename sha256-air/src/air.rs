use crate::columns::{Sha256AuxCols, Sha256IoCols, Word};

use super::columns::Sha256Cols;
use afs_chips::bits::bit_decompose::air::BitDecomposeAir;
use afs_chips::bits::bit_decompose::columns::BitDecomposeCols;
use afs_chips::sub_chip::{AirConfig, SubAir};
use p3_air::{Air, AirBuilder, BaseAir, FilteredAirBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::borrow::Borrow;

// TODO: update
pub const NUM_COLUMNS: usize = 100;

pub const H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub struct Sha256Air {
    bus_index: usize,
}

impl AirConfig for Sha256Air {
    type Cols<T> = Sha256Cols<T>;
}

impl<F: Field> BaseAir<F> for Sha256Air {
    fn width(&self) -> usize {
        NUM_COLUMNS
    }
}

impl<AB: AirBuilder> Air<AB> for Sha256Air {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &Sha256Cols<AB::Var> = (*local).borrow();
        let next: &Sha256Cols<AB::Var> = (*next).borrow();

        SubAir::<AB>::eval(self, builder, local.io, (local.aux, next.aux));
    }
}

impl<AB: AirBuilder> SubAir<AB> for Sha256Air {
    type IoView = Sha256IoCols<AB::Var>;
    // local, next
    type AuxView = (Sha256AuxCols<AB::Var>, Sha256AuxCols<AB::Var>);

    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        self.eval_init(builder, &aux.0);
        self.eval_compression(builder, &io, &aux.0, &aux.1);
    }
}

impl Sha256Air {
    fn eval_io<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: &Sha256IoCols<AB::Var>,
        aux: &Sha256AuxCols<AB::Var>,
    ) {
        todo!()
    }

    fn eval_init<AB: AirBuilder>(&self, builder: &mut AB, aux: &Sha256AuxCols<AB::Var>) {
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h0, H[0]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h1, H[1]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h2, H[2]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h3, H[3]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h4, H[4]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h5, H[5]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h6, H[6]);
        self.assert_word_equal_u32(&mut builder.when_first_row(), &aux.h7, H[7]);
    }

    fn eval_compression<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local_io: &Sha256IoCols<AB::Var>,
        local_aux: &Sha256AuxCols<AB::Var>,
        next_aux: &Sha256AuxCols<AB::Var>,
    ) {
        // Working variables set to hashes at the start of each block.
        let mut block_start = builder.when(local_aux.is_block_start);
        self.assert_words_equal(&mut block_start, &local_aux.a, &next_aux.h0);
        self.assert_words_equal(&mut block_start, &local_aux.b, &next_aux.h1);
        self.assert_words_equal(&mut block_start, &local_aux.c, &next_aux.h2);
        self.assert_words_equal(&mut block_start, &local_aux.d, &next_aux.h3);
        self.assert_words_equal(&mut block_start, &local_aux.e, &next_aux.h4);
        self.assert_words_equal(&mut block_start, &local_aux.f, &next_aux.h5);
        self.assert_words_equal(&mut block_start, &local_aux.g, &next_aux.h6);
        self.assert_words_equal(&mut block_start, &local_aux.h, &next_aux.h7);

        let mut is_main_loop = builder.when_ne(local_aux.row_idx, AB::F::zero());
        let bit_decompose_air = BitDecomposeAir::<32>::new(self.bus_index);

        let x = std::iter::once(local_aux.e).chain(local_aux.e_bits);
        SubAir::eval(
            &bit_decompose_air,
            builder,
            BitDecomposeCols::from_slice(std::iter::once(local_aux.e).chain(local_aux.e_bits)),
            (),
        );

        // Update the working variables for next round.

        // Update hashes at the end of each block.
    }

    fn assert_word_equal_u32<AB: AirBuilder>(
        &self,
        builder: &mut FilteredAirBuilder<'_, AB>,
        word: &Word<AB::Var>,
        x: u32,
    ) {
        let high = AB::F::from_canonical_u16((x >> 16) as u16);
        let low = AB::F::from_canonical_u16((x & 0xffff) as u16);
        builder.assert_eq(word.0[0], high);
        builder.assert_eq(word.0[1], low);
    }

    fn assert_words_equal<AB: AirBuilder>(
        &self,
        builder: &mut FilteredAirBuilder<'_, AB>,
        w1: &Word<AB::Var>,
        w2: &Word<AB::Var>,
    ) {
        builder.assert_eq(w1.0[0], w2.0[0]);
        builder.assert_eq(w1.0[1], w2.0[1]);
    }
}
