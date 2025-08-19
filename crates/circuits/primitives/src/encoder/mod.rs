use std::ops::RangeInclusive;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{Field, FieldAlgebra},
};

use crate::SubAir;

#[cfg(all(test, feature = "cuda"))]
mod tests;

/// Efficient encoding of circuit selectors
///
/// This encoder represents selectors as points in a k-dimensional space where each
/// coordinate is between 0 and max_degree, and their sum doesn't exceed max_degree.
/// This approach allows encoding many selectors with significantly fewer columns
/// than the traditional approach of using one boolean column per selector.
#[derive(Clone, Debug)]
pub struct Encoder {
    /// Number of variables (columns) used to encode the flags
    var_cnt: usize,
    /// The number of flags, excluding the invalid/dummy flag.
    flag_cnt: usize,
    /// Maximal degree of the flag expressions.
    /// The maximal degree of the equalities in the AIR, however, **is one higher:** that is,
    /// `max_flag_degree + 1`.
    max_flag_degree: u32,
    /// All possible points in the k-dimensional space that can be used to encode flags
    pts: Vec<Vec<u32>>,
    /// Whether the zero point (0,...,0) is reserved for invalid/dummy rows
    reserve_invalid: bool,
}

impl Encoder {
    /// Create a new encoder for a given number of flags and maximum degree.
    /// The flags will correspond to points in F^k, where k is the number of variables.
    /// The zero point is reserved for the dummy row.
    /// `max_degree` is the upper bound for the flag expressions, but the `eval` function
    /// of the encoder itself will use some constraints of degree `max_degree + 1`.
    /// `reserve_invalid` indicates if the encoder should reserve the (0, ..., 0) point as an
    /// invalid/dummy flag.
    pub fn new(cnt: usize, max_degree: u32, reserve_invalid: bool) -> Self {
        // Calculate binomial coefficient (d+k choose k) to determine how many points we can encode
        let binomial = |x: u32| {
            let mut res = 1;
            for i in 1..=max_degree {
                res = res * (x + i) / i;
            }
            res
        };
        // Find minimum k (number of variables) needed to encode cnt flags
        let k = (0..)
            .find(|&x| binomial(x) >= cnt as u32 + reserve_invalid as u32)
            .unwrap() as usize;

        // Generate all points where coordinates sum to at most max_degree
        let mut cur = vec![0u32; k];
        let mut sum = 0;
        let mut pts = Vec::new();
        loop {
            pts.push(cur.clone());
            if cur[0] == max_degree {
                break;
            }
            let mut i = k - 1;
            while sum == max_degree {
                sum -= cur[i];
                cur[i] = 0;
                i -= 1;
            }
            sum += 1;
            cur[i] += 1;
        }
        Self {
            var_cnt: k,
            flag_cnt: cnt,
            max_flag_degree: max_degree,
            pts,
            reserve_invalid,
        }
    }

    /// Construct the multivariate Lagrange polynomial for a specific point
    /// This polynomial equals 1 at the given point and 0 at all other points
    /// in our solution set
    fn expression_for_point<AB: InteractionBuilder>(
        &self,
        pt: &[u32],
        vars: &[AB::Var],
    ) -> AB::Expr {
        assert_eq!(self.var_cnt, pt.len(), "wrong point dimension");
        assert_eq!(self.var_cnt, vars.len(), "wrong number of variables");
        let mut expr = AB::Expr::ONE;
        let mut denom = AB::F::ONE;

        // First part: product for each coordinate
        for (i, &coord) in pt.iter().enumerate() {
            for j in 0..coord {
                expr *= vars[i] - AB::Expr::from_canonical_u32(j);
                denom *= AB::F::from_canonical_u32(coord - j);
            }
        }

        // Second part: ensure the sum doesn't exceed max_degree
        {
            let sum: u32 = pt.iter().sum();
            let var_sum = vars.iter().fold(AB::Expr::ZERO, |acc, &v| acc + v);
            for j in 0..(self.max_flag_degree - sum) {
                expr *= AB::Expr::from_canonical_u32(self.max_flag_degree - j) - var_sum.clone();
                denom *= AB::F::from_canonical_u32(j + 1);
            }
        }
        expr * denom.inverse()
    }

    /// Get the polynomial expression that equals 1 when the variables encode the flag at index
    /// flag_idx
    pub fn get_flag_expr<AB: InteractionBuilder>(
        &self,
        flag_idx: usize,
        vars: &[AB::Var],
    ) -> AB::Expr {
        assert!(flag_idx < self.flag_cnt, "flag index out of range");
        self.expression_for_point::<AB>(&self.pts[flag_idx + self.reserve_invalid as usize], vars)
    }

    /// Get the point coordinates that correspond to the flag at index flag_idx
    pub fn get_flag_pt(&self, flag_idx: usize) -> Vec<u32> {
        assert!(flag_idx < self.flag_cnt, "flag index out of range");
        self.pts[flag_idx + self.reserve_invalid as usize].clone()
    }

    /// Returns an expression that is 1 if the variables encode a valid flag and 0 if they encode
    /// the invalid point
    pub fn is_valid<AB: InteractionBuilder>(&self, vars: &[AB::Var]) -> AB::Expr {
        AB::Expr::ONE - self.expression_for_point::<AB>(&self.pts[0], vars)
    }

    /// Returns all flag expressions for the given variables
    pub fn flags<AB: InteractionBuilder>(&self, vars: &[AB::Var]) -> Vec<AB::Expr> {
        (0..self.flag_cnt)
            .map(|i| self.get_flag_expr::<AB>(i, vars))
            .collect()
    }

    /// Returns the sum of expressions for all unused points
    /// This is used to ensure that variables encode only valid flags
    pub fn sum_of_unused<AB: InteractionBuilder>(&self, vars: &[AB::Var]) -> AB::Expr {
        let mut expr = AB::Expr::ZERO;
        for i in (self.flag_cnt + self.reserve_invalid as usize)..self.pts.len() {
            expr += self.expression_for_point::<AB>(&self.pts[i], vars);
        }
        expr
    }

    /// Returns the number of variables used for encoding
    pub fn width(&self) -> usize {
        self.var_cnt
    }

    /// Returns an expression that is 1 if `flag_idxs` contains the encoded flag and 0 otherwise
    pub fn contains_flag<AB: InteractionBuilder>(
        &self,
        vars: &[AB::Var],
        flag_idxs: &[usize],
    ) -> AB::Expr {
        flag_idxs.iter().fold(AB::Expr::ZERO, |acc, flag_idx| {
            acc + self.get_flag_expr::<AB>(*flag_idx, vars)
        })
    }

    /// Returns an expression that is 1 if (l..=r) contains the encoded flag and 0 otherwise
    pub fn contains_flag_range<AB: InteractionBuilder>(
        &self,
        vars: &[AB::Var],
        range: RangeInclusive<usize>,
    ) -> AB::Expr {
        self.contains_flag::<AB>(vars, &range.collect::<Vec<_>>())
    }

    /// Returns an expression that is 0 if `flag_idxs_vals` doesn't contain the encoded flag
    /// and the corresponding val if it does
    /// `flag_idxs_vals` is a list of tuples (flag_idx, val)
    pub fn flag_with_val<AB: InteractionBuilder>(
        &self,
        vars: &[AB::Var],
        flag_idx_vals: &[(usize, usize)],
    ) -> AB::Expr {
        flag_idx_vals
            .iter()
            .fold(AB::Expr::ZERO, |acc, (flag_idx, val)| {
                acc + self.get_flag_expr::<AB>(*flag_idx, vars)
                    * AB::Expr::from_canonical_usize(*val)
            })
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for Encoder {
    type AirContext<'a>
        = &'a [AB::Var]
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, local: &'a [AB::Var])
    where
        AB: 'a,
        AB::Expr: 'a,
    {
        assert_eq!(local.len(), self.var_cnt, "wrong number of variables");

        // Helper function to create the product (x-0)(x-1)...(x-max_degree)
        let falling_factorial = |lin: AB::Expr| {
            let mut res = AB::Expr::ONE;
            for i in 0..=self.max_flag_degree {
                res *= lin.clone() - AB::Expr::from_canonical_u32(i);
            }
            res
        };
        // All x_i are from 0 to max_degree
        for &var in local.iter() {
            builder.assert_zero(falling_factorial(var.into()))
        }
        // Sum of all x_i is from 0 to max_degree
        builder.assert_zero(falling_factorial(
            local.iter().fold(AB::Expr::ZERO, |acc, &x| acc + x),
        ));
        // This constraint guarantees that the encoded point either:
        // 1. Is the zero point (0,...,0) if reserved for invalid/dummy rows, or
        // 2. Represents one of our defined selectors (flag_idx from 0 to flag_cnt-1)
        // It works by requiring the sum of Lagrange polynomials for all unused points to be zero,
        // which forces the current point to be one of our explicitly defined selector patterns
        builder.assert_zero(self.sum_of_unused::<AB>(local));
    }
}
