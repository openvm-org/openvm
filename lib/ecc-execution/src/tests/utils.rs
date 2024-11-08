use axvm_ecc::{
    field::{Field, FieldExt},
    point::{AffineCoords, EcPoint},
};
use group::ScalarMul;
use itertools::izip;
use rand::{rngs::StdRng, SeedableRng};

/// Generates a set of random G1 and G2 points from a random seed and outputs the vectors of P and Q points as well as
/// the corresponding P and Q EcPoint structs.
#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points<A1, A2, Fp, Fp2>(
    rand_seeds: &[u64],
) -> (Vec<A1>, Vec<A2>, Vec<EcPoint<Fp>>, Vec<EcPoint<Fp2>>)
where
    A1: AffineCoords<Fp>,
    A2: AffineCoords<Fp2>,
    Fp: Field,
    Fp2: FieldExt<BaseField = Fp>,
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

/// Generates test points for N number of points for an elliptic curve pairing, where the inputs `a` and `b` are
/// scalars of generators in G1 and G2, respectively. Importantly, for every even index, the generator P point is
/// negated (reflected an the x-axis). Outputs the vectors of P and Q points as well as the corresponding P and Q
/// EcPoint structs.
#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points_generator_scalar<A1, A2, Fr, Fp, Fp2, const N: usize>(
    a: &[Fr; N],
    b: &[Fr; N],
) -> (Vec<A1>, Vec<A2>, Vec<EcPoint<Fp>>, Vec<EcPoint<Fp2>>)
where
    A1: AffineCoords<Fp> + ScalarMul<Fr>,
    A2: AffineCoords<Fp2> + ScalarMul<Fr>,
    Fr: Field,
    Fp: Field,
    Fp2: FieldExt<BaseField = Fp>,
{
    assert!(N % 2 == 0, "Must have even number of P and Q scalars");
    let mut P_vec = vec![];
    let mut Q_vec = vec![];
    for i in 0..N {
        let mut p = A1::generator() * a[i].clone();
        if i % 2 == 1 {
            p = p.neg();
        }
        let q = A2::generator() * b[i].clone();
        P_vec.push(p);
        Q_vec.push(q);
    }
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
