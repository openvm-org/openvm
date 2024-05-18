use columns::XorCols;
use p3_air::AirBuilder;
use p3_field::AbstractField;

use afs_stark_backend::interaction::Interaction;
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use self::columns::XorIOCols;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct XorChip<const N: usize> {
    bus_index: usize,

    pub pairs: Vec<(u32, u32)>,
}

impl<const N: usize> XorChip<N> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self { bus_index, pairs }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    fn calc_xor(&self, a: u32, b: u32) -> u32 {
        a ^ b
    }

    pub fn request(&mut self, a: u32, b: u32) -> u32 {
        self.pairs.push((a, b));
        self.calc_xor(a, b)
    }

    /// Imposes AIR constraints within each row of the trace
    /// Constraints x, y, z to be equal to their bit representation in x_bits, y_bits, z_bits.
    /// For each x_bit[i], y_bit[i], and z_bit[i], constraints x_bit[i] + y_bit[i] - 2 * x_bit[i] * y_bit[i] == z_bit[i],
    /// which is equivalent to ensuring that x_bit[i] ^ y_bit[i] == z_bit[i].
    /// Overall, this ensures that x^y == z.
    pub fn impose_constraints<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        xor_cols: XorCols<N, AB::Var>,
    ) where
        AB::Var: Clone,
    {
        let mut x_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            x_from_bits += xor_cols.x_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(x_from_bits, xor_cols.io.x);

        let mut y_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            y_from_bits += xor_cols.y_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(y_from_bits, xor_cols.io.y);

        let mut z_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            z_from_bits += xor_cols.z_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(z_from_bits, xor_cols.io.z);

        for i in 0..N {
            builder.assert_eq(
                xor_cols.x_bits[i] + xor_cols.y_bits[i]
                    - AB::Expr::two() * xor_cols.x_bits[i] * xor_cols.y_bits[i],
                xor_cols.z_bits[i],
            );
        }
    }

    pub fn receives_custom<F: PrimeField64>(&self, cols: XorIOCols<usize>) -> Interaction<F> {
        Interaction {
            fields: vec![
                VirtualPairCol::single_main(cols.x),
                VirtualPairCol::single_main(cols.y),
                VirtualPairCol::single_main(cols.z),
            ],
            count: VirtualPairCol::constant(F::one()),
            argument_index: self.bus_index(),
        }
    }
}
