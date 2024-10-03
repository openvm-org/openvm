use halo2curves_axiom::ff::Field;

use crate::common::FieldExtension;

pub fn final_exp_hint<Fp, Fp2, Fp12>(_f: Fp12) -> (Fp12, Fp12)
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    // f = c^λ * u
    unimplemented!("final_exp_hint is not implemented");
}

pub fn final_exponentiation<Fp, Fp2, Fp12>(_f: Fp12) -> Fp12
where
    Fp: Field,
    Fp2: FieldExtension<BaseField = Fp>,
    Fp12: FieldExtension<BaseField = Fp2>,
{
    // func FinalExponentiation(z *GT, _z ...*GT) GT {
    // 	var result GT
    // 	result.Set(z)

    // 	var t [3]GT

    // Easy part
    // (p⁶-1)(p²+1)
    // 	t[0].Conjugate(&result)
    // 	result.Inverse(&result)
    // 	t[0].Mul(&t[0], &result)
    // 	result.FrobeniusSquare(&t[0]).
    // 		Mul(&result, &t[0])
    // t[0].conjugate();
    // t[0] = t[0].invert().unwrap();
    // t[0].square();

    // 	var one GT
    // 	one.SetOne()
    // 	if result.Equal(&one) {
    // 		return result
    // 	}

    // 	// Hard part (up to permutation)
    // 	// Daiki Hayashida, Kenichiro Hayasaka and Tadanori Teruya
    // 	// https://eprint.iacr.org/2020/875.pdf
    // 	t[0].CyclotomicSquare(&result)
    // 	t[1].ExptHalf(&t[0])
    // 	t[2].InverseUnitary(&result)
    // 	t[1].Mul(&t[1], &t[2])
    // 	t[2].Expt(&t[1])
    // 	t[1].InverseUnitary(&t[1])
    // 	t[1].Mul(&t[1], &t[2])
    // 	t[2].Expt(&t[1])
    // 	t[1].Frobenius(&t[1])
    // 	t[1].Mul(&t[1], &t[2])
    // 	result.Mul(&result, &t[0])
    // 	t[0].Expt(&t[1])
    // 	t[2].Expt(&t[0])
    // 	t[0].FrobeniusSquare(&t[1])
    // 	t[1].InverseUnitary(&t[1])
    // 	t[1].Mul(&t[1], &t[2])
    // 	t[1].Mul(&t[1], &t[0])
    // 	result.Mul(&result, &t[1])

    // 	return result
    // }
    unimplemented!("final_exponentiation is not implemented");
}
