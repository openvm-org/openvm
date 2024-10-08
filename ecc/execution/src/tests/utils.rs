use halo2curves_axiom::{ff::Field, group::Group};
use itertools::izip;
use rand::{rngs::StdRng, SeedableRng};

use crate::common::{AffineCoords, EcPoint, FieldExtension};

#[allow(non_snake_case)]
pub fn generate_test_points<A1, A2, Fp, Fp2>(
    rand_seeds: &[u64],
) -> (Vec<A1>, Vec<A2>, Vec<EcPoint<Fp>>, Vec<EcPoint<Fp2>>)
where
    A1: AffineCoords<Fp>,
    A2: AffineCoords<Fp2>,
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
{
    let (P_vec, Q_vec) = rand_seeds
        .iter()
        .map(|seed| {
            let mut rng0 = StdRng::seed_from_u64(*seed);
            let p = A1::random(&mut rng0);
            let mut rng1 = StdRng::seed_from_u64(*seed * 2);
            let q = A2::random(&mut rng1);
            (p, q)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let (P_ecpoints, Q_ecpoints) = izip!(P_vec.clone(), Q_vec.clone())
        .map(|(P, Q)| {
            (
                EcPoint { x: P.x(), y: P.y() },
                EcPoint { x: Q.x(), y: Q.y() },
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    (P_vec, Q_vec, P_ecpoints, Q_ecpoints)
}
