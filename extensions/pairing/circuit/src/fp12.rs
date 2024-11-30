use std::{array::from_fn, cell::RefCell, rc::Rc};

use ax_mod_circuit_builder::{ExprBuilder, FieldVariable};
use axvm_mod_circuit::Fp2;

/// Field extension Fp12 defined with coefficients in Fp2.
/// Represents the element `c0 + c1 w + ... + c5 w^5` in Fp12.
/// Fp6-equivalent coefficients are c0: (c0, c2, c4), c1: (c1, c3, c5).
pub struct Fp12 {
    pub c: [Fp2; 6],
}

impl Fp12 {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c = from_fn(|_| Fp2::new(builder.clone()));
        Fp12 { c }
    }

    pub fn save(&mut self) -> [usize; 12] {
        self.c
            .each_mut()
            .map(|c| c.save())
            .concat()
            .try_into()
            .unwrap()
    }

    pub fn save_output(&mut self) {
        for c in self.c.iter_mut() {
            c.save_output();
        }
    }

    pub fn add(&mut self, other: &mut Fp12) -> Fp12 {
        Fp12 {
            c: from_fn(|i| self.c[i].add(&mut other.c[i])),
        }
    }

    pub fn sub(&mut self, other: &mut Fp12) -> Fp12 {
        Fp12 {
            c: from_fn(|i| self.c[i].sub(&mut other.c[i])),
        }
    }

    pub fn mul(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp12 {
        let c = from_fn(|i| {
            let mut sum = self.c[0].mul(&mut other.c[i]);
            for j in 1..=5.min(i) {
                let k = i - j;
                sum = sum.add(&mut self.c[j].mul(&mut other.c[k]));
            }
            let mut sum_hi: Option<Fp2> = None;
            for j in (i + 1)..=5 {
                let k = 6 + i - j;
                let mut term = self.c[j].mul(&mut other.c[k]);
                if let Some(mut running_sum) = sum_hi {
                    sum_hi = Some(running_sum.add(&mut term));
                } else {
                    sum_hi = Some(term);
                }
            }
            if let Some(mut sum_hi) = sum_hi {
                sum = sum.add(&mut sum_hi.int_mul(xi));
            }
            sum.save();
            sum
        });
        Fp12 { c }
    }

    /// Multiply self by `x0 + x1 w + x2 w^2 + x3 w^3 + x4 w^4` in Fp12.
    pub fn mul_by_01234(
        &mut self,
        x0: &mut Fp2,
        x1: &mut Fp2,
        x2: &mut Fp2,
        x3: &mut Fp2,
        x4: &mut Fp2,
        xi: [isize; 2],
    ) -> Fp12 {
        let c0 = self.c[0].mul(x0).add(
            &mut self.c[2]
                .mul(x4)
                .add(&mut self.c[3].mul(x3))
                .add(&mut self.c[4].mul(x2))
                .add(&mut self.c[5].mul(x1))
                .int_mul(xi),
        );

        let c1 = self.c[0].mul(x1).add(&mut self.c[1].mul(x0)).add(
            &mut self.c[3]
                .mul(x4)
                .add(&mut self.c[4].mul(x3))
                .add(&mut self.c[5].mul(x2))
                .int_mul(xi),
        );

        let c2 = self.c[0]
            .mul(x2)
            .add(&mut self.c[1].mul(x1))
            .add(&mut self.c[2].mul(x0))
            .add(&mut self.c[4].mul(x4).add(&mut self.c[5].mul(x3)).int_mul(xi));

        let c3 = self.c[0]
            .mul(x3)
            .add(&mut self.c[1].mul(x2))
            .add(&mut self.c[2].mul(x1))
            .add(&mut self.c[3].mul(x0))
            .add(&mut self.c[5].mul(x4).int_mul(xi));

        let c4 = self.c[0]
            .mul(x4)
            .add(&mut self.c[1].mul(x3))
            .add(&mut self.c[2].mul(x2))
            .add(&mut self.c[3].mul(x1))
            .add(&mut self.c[4].mul(x0));

        let c5 = self.c[1]
            .mul(x4)
            .add(&mut self.c[2].mul(x3))
            .add(&mut self.c[3].mul(x2))
            .add(&mut self.c[4].mul(x1))
            .add(&mut self.c[5].mul(x0));

        Fp12 {
            c: [c0, c1, c2, c3, c4, c5],
        }
    }

    /// Multiply `self` by `x0 + x2 w^2 + x3 w^3 + x4 w^4 + x5 w^5` in Fp12.
    pub fn mul_by_02345(
        &mut self,
        x0: &mut Fp2,
        x2: &mut Fp2,
        x3: &mut Fp2,
        x4: &mut Fp2,
        x5: &mut Fp2,
        xi: [isize; 2],
    ) -> Fp12 {
        let c0 = self.c[0].mul(x0).add(
            &mut self.c[1]
                .mul(x5)
                .add(&mut self.c[2].mul(x4))
                .add(&mut self.c[3].mul(x3))
                .add(&mut self.c[4].mul(x2))
                .int_mul(xi),
        );

        let c1 = self.c[1].mul(x0).add(
            &mut self.c[2]
                .mul(x5)
                .add(&mut self.c[3].mul(x4))
                .add(&mut self.c[4].mul(x3))
                .add(&mut self.c[5].mul(x2))
                .int_mul(xi),
        );

        let c2 = self.c[0].mul(x2).add(&mut self.c[2].mul(x0)).add(
            &mut self.c[3]
                .mul(x5)
                .add(&mut self.c[4].mul(x4))
                .add(&mut self.c[5].mul(x3))
                .int_mul(xi),
        );

        let c3 = self.c[0]
            .mul(x3)
            .add(&mut self.c[1].mul(x2))
            .add(&mut self.c[3].mul(x0))
            .add(&mut self.c[4].mul(x5).add(&mut self.c[5].mul(x4)).int_mul(xi));

        let c4 = self.c[0]
            .mul(x4)
            .add(&mut self.c[1].mul(x3))
            .add(&mut self.c[2].mul(x2))
            .add(&mut self.c[4].mul(x0))
            .add(&mut self.c[5].mul(x5).int_mul(xi));

        let c5 = self.c[0]
            .mul(x5)
            .add(&mut self.c[1].mul(x4))
            .add(&mut self.c[2].mul(x3))
            .add(&mut self.c[3].mul(x2))
            .add(&mut self.c[5].mul(x0));

        Fp12 {
            c: [c0, c1, c2, c3, c4, c5],
        }
    }

    pub fn div(&mut self, _other: &mut Fp12, _xi: [isize; 2]) -> Fp12 {
        unimplemented!()
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable) -> Fp12 {
        Fp12 {
            c: from_fn(|i| self.c[i].scalar_mul(fp)),
        }
    }
}

#[cfg(test)]
mod tests {
    use ax_circuit_primitives::TraceSubRowGenerator;
    use ax_ecc_execution::axvm_ecc::algebra::field::FieldExtension;
    use ax_mod_circuit_builder::{test_utils::*, *};
    use ax_stark_backend::{
        p3_air::BaseAir, p3_field::AbstractField, p3_matrix::dense::RowMajorMatrix,
    };
    use ax_stark_sdk::{
        any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
        p3_baby_bear::BabyBear, utils::create_seeded_rng,
    };
    use axvm_ecc_constants::BN254;
    use halo2curves_axiom::{bn256::Fq12, ff::Field};

    use super::*;

    fn generate_random_fq12() -> Fq12 {
        let mut rng = create_seeded_rng();
        Fq12::random(&mut rng)
    }

    fn run_fp12_test(
        x: Fq12,
        y: Fq12,
        xi: Option<[isize; 2]>,
        fp12_fn_addsub: Option<impl Fn(&mut Fp12, &mut Fp12) -> Fp12>,
        fp12_fn_mul: Option<impl Fn(&mut Fp12, &mut Fp12, [isize; 2]) -> Fp12>,
        fq12_fn: impl Fn(&Fq12, &Fq12) -> Fq12,
    ) {
        if fp12_fn_addsub.is_none() && fp12_fn_mul.is_none() {
            panic!("Either fp12_fn_addsub or fp12_fn_mul must be provided");
        }
        if fp12_fn_addsub.is_some() && fp12_fn_mul.is_some() {
            panic!("Only one of fp12_fn_addsub or fp12_fn_mul must be provided");
        }

        let prime = BN254.MODULUS.clone();
        let (range_checker, builder) = setup(&prime);

        let mut x_fp12 = Fp12::new(builder.clone());
        let mut y_fp12 = Fp12::new(builder.clone());
        let mut r = if let Some(fp12_fn_addsub) = fp12_fn_addsub {
            fp12_fn_addsub(&mut x_fp12, &mut y_fp12)
        } else {
            let fp12_fn_mul = fp12_fn_mul.unwrap();
            fp12_fn_mul(&mut x_fp12, &mut y_fp12, xi.unwrap())
        };
        let indices = r.save();

        let builder = builder.borrow().clone();
        let air = FieldExpr::new(builder, range_checker.bus(), false);
        let width = BaseAir::<BabyBear>::width(&air);

        let x_fq12 = x;
        let y_fq12 = y;
        let r_fq12 = fq12_fn(&x_fq12, &y_fq12);
        let mut inputs = bn254_fq12_to_biguint_vec(x_fq12);
        inputs.extend(bn254_fq12_to_biguint_vec(y_fq12));

        let mut row = BabyBear::zero_vec(width);
        air.generate_subrow((&range_checker, inputs, vec![]), &mut row);
        let FieldExprCols { vars, .. } = air.load_vars(&row);
        let trace = RowMajorMatrix::new(row, width);
        let range_trace = range_checker.generate_trace();

        for (idx, v) in indices
            .iter()
            .zip(r_fq12.to_coeffs().into_iter().flat_map(|x| [x.c0, x.c1]))
        {
            assert_eq!(
                evaluate_biguint(&vars[*idx], LIMB_BITS),
                bn254_fq_to_biguint(v)
            );
        }

        BabyBearBlake3Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![air, range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }

    #[test]
    fn test_fp12_add() {
        let x = generate_random_fq12();
        let y = generate_random_fq12();
        run_fp12_test(
            x,
            y,
            None,
            Some(Fp12::add),
            None::<fn(&mut Fp12, &mut Fp12, [isize; 2]) -> Fp12>,
            |x, y| x + y,
        );
    }

    #[test]
    fn test_fp12_sub() {
        let x = generate_random_fq12();
        let y = generate_random_fq12();
        run_fp12_test(
            x,
            y,
            None,
            Some(Fp12::sub),
            None::<fn(&mut Fp12, &mut Fp12, [isize; 2]) -> Fp12>,
            |x, y| x - y,
        );
    }

    #[test]
    fn test_fp12_mul() {
        let x = generate_random_fq12();
        let y = generate_random_fq12();
        let xi = [9, 1];
        run_fp12_test(
            x,
            y,
            Some(xi),
            None::<fn(&mut Fp12, &mut Fp12) -> Fp12>,
            Some(Fp12::mul),
            |x, y| x * y,
        );
    }
}
