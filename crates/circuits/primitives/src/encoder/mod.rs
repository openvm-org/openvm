use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{AbstractField, Field},
};

use crate::SubAir;

#[derive(Clone, Debug)]
pub struct Encoder {
    var_cnt: usize,
    flag_cnt: usize,
    /// Maximal degree of the flag expressions.
    /// The maximal degree of the equalities in the AIR, however, **is one higher:** that is, `max_flag_degree + 1`.
    max_flag_degree: u32,
    pts: Vec<Vec<u32>>,
}

impl Encoder {
    /// Create a new encoder for a given number of flags and maximum degree.
    /// The flags will correspond to points in F^k, where k is the number of variables.
    /// The zero point is reserved for the dummy row.
    /// `max_degree` is the upper bound for the flag expressions, but the `eval` function
    /// of the encoder itself will use some constraints of degree `max_degree + 1`.
    pub fn new(cnt: usize, max_degree: u32) -> Self {
        let binomial = |x: u32| {
            let mut res = 1;
            for i in 1..=max_degree {
                res = res * (x + i) / i;
            }
            res
        };
        let k = (0..).find(|&x| binomial(x) > cnt as u32).unwrap() as usize;
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
        self.expression_for_point::<AB>(&self.pts[flag_idx + 1], vars)
    }

    pub fn get_flag_pt(&self, flag_idx: usize) -> Vec<u32> {
        assert!(flag_idx < self.flag_cnt, "flag index out of range");
        self.pts[flag_idx + 1].clone()
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
        for i in self.flag_cnt + 1..self.pts.len() {
            expr += self.expression_for_point::<AB>(&self.pts[i], vars);
        }
        expr
    }

    pub fn width(&self) -> usize {
        self.var_cnt
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
