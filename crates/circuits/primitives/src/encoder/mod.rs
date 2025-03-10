use std::ops::RangeInclusive;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{Field, FieldAlgebra},
};

use crate::SubAir;

#[derive(Clone, Debug)]
pub struct Encoder {
    var_cnt: usize,
    /// The number of flags, excluding the invalid/dummy flag.
    flag_cnt: usize,
    /// Maximal degree of the flag expressions.
    /// The maximal degree of the equalities in the AIR, however, **is one higher:** that is, `max_flag_degree + 1`.
    max_flag_degree: u32,
    pts: Vec<Vec<u32>>,
    reserve_invalid: bool,
}

impl Encoder {
    /// Create a new encoder for a given number of flags and maximum degree.
    /// The flags will correspond to points in F^k, where k is the number of variables.
    /// The zero point is reserved for the dummy row.
    /// `max_degree` is the upper bound for the flag expressions, but the `eval` function
    /// of the encoder itself will use some constraints of degree `max_degree + 1`.
    /// `reserve_invalid` indicates if the encoder should reserve the (0, ..., 0) point as an invalid/dummy flag.
    pub fn new(cnt: usize, max_degree: u32, reserve_invalid: bool) -> Self {
        let binomial = |x: u32| {
            let mut res = 1;
            for i in 1..=max_degree {
                res = res * (x + i) / i;
            }
            res
        };
        let k = (0..)
            .find(|&x| binomial(x) >= cnt as u32 + reserve_invalid as u32)
            .unwrap() as usize;
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

    fn expression_for_point<AB: InteractionBuilder>(
        &self,
        pt: &[u32],
        vars: &[AB::Var],
    ) -> AB::Expr {
        assert_eq!(self.var_cnt, pt.len(), "wrong point dimension");
        assert_eq!(self.var_cnt, vars.len(), "wrong number of variables");
        let mut expr = AB::Expr::ONE;
        let mut denom = AB::F::ONE;
        for (i, &coord) in pt.iter().enumerate() {
            for j in 0..coord {
                expr *= vars[i] - AB::Expr::from_canonical_u32(j);
                denom *= AB::F::from_canonical_u32(coord - j);
            }
        }
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

    pub fn get_flag_expr<AB: InteractionBuilder>(
        &self,
        flag_idx: usize,
        vars: &[AB::Var],
    ) -> AB::Expr {
        assert!(flag_idx < self.flag_cnt, "flag index out of range");
        self.expression_for_point::<AB>(&self.pts[flag_idx + self.reserve_invalid as usize], vars)
    }

    pub fn get_flag_pt(&self, flag_idx: usize) -> Vec<u32> {
        assert!(flag_idx < self.flag_cnt, "flag index out of range");
        self.pts[flag_idx + self.reserve_invalid as usize].clone()
    }

    pub fn is_valid<AB: InteractionBuilder>(&self, vars: &[AB::Var]) -> AB::Expr {
        AB::Expr::ONE - self.expression_for_point::<AB>(&self.pts[0], vars)
    }

    pub fn flags<AB: InteractionBuilder>(&self, vars: &[AB::Var]) -> Vec<AB::Expr> {
        (0..self.flag_cnt)
            .map(|i| self.get_flag_expr::<AB>(i, vars))
            .collect()
    }

    pub fn sum_of_unused<AB: InteractionBuilder>(&self, vars: &[AB::Var]) -> AB::Expr {
        let mut expr = AB::Expr::ZERO;
        for i in (self.flag_cnt + self.reserve_invalid as usize)..self.pts.len() {
            expr += self.expression_for_point::<AB>(&self.pts[i], vars);
        }
        expr
    }

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
        // Either all x_i are zero, or this point corresponds to some flag
        builder.assert_zero(self.sum_of_unused::<AB>(local));
    }
}
