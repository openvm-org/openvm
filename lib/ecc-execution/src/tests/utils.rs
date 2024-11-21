use axvm_ecc::{algebra::field::FieldExtension, AffineCoords, AffinePoint};
use ff::Field;
use itertools::izip;
use rand::{rngs::StdRng, SeedableRng};

/// Generates a set of random G1 and G2 points from a random seed and outputs the vectors of P and Q points as well as
/// the corresponding P and Q EcPoint structs.
#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points<A1, A2, Fp, Fp2>(
    rand_seeds: &[u64],
) -> (
    Vec<A1>,
    Vec<A2>,
    Vec<AffinePoint<Fp>>,
    Vec<AffinePoint<Fp2>>,
)
where
    A1: AffineCoords<Fp>,
    A2: AffineCoords<Fp2>,
    Fp: Field,
    Fp2: FieldExtension<Fp>,
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
                AffinePoint { x: P.x(), y: P.y() },
                AffinePoint { x: Q.x(), y: Q.y() },
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
pub fn generate_test_points_generator_scalar<A1, A2, Fp, Fp2, const N: usize>(
    a: &[i32; N],
    b: &[i32; N],
) -> (
    Vec<A1>,
    Vec<A2>,
    Vec<AffinePoint<Fp>>,
    Vec<AffinePoint<Fp2>>,
)
where
    A1: AffineCoords<Fp>,
    A2: AffineCoords<Fp2>,
    // Fr: Field,
    Fp: Field,
    Fp2: Field + FieldExtension<Fp>,
{
    assert!(N % 2 == 0, "Must have even number of P and Q scalars");
    let mut P_vec: Vec<A1> = vec![];
    let mut Q_vec: Vec<A2> = vec![];
    for i in 0..N {
        let p = A1::generator();
        let p_mul: A1 = if a[i].is_negative() {
            A1::new(Fp::ONE * p.x(), Fp::ONE.neg() * p.y())
        } else {
            A1::new(Fp::ONE * p.x(), Fp::ONE * p.y())
        };
        let q = A2::generator();
        let q_mul: A2 = if b[i].is_negative() {
            A2::new(Fp2::ONE * q.x(), Fp2::ONE.neg() * q.y())
        } else {
            A2::new(Fp2::ONE * q.x(), Fp2::ONE * q.y())
        };
        P_vec.push(p_mul);
        Q_vec.push(q_mul);
    }
    let (P_ecpoints, Q_ecpoints) = izip!(P_vec.clone(), Q_vec.clone())
        .map(|(P, Q)| {
            (
                AffinePoint { x: P.x(), y: P.y() },
                AffinePoint { x: Q.x(), y: Q.y() },
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    (P_vec, Q_vec, P_ecpoints, Q_ecpoints)
}
