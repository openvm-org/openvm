pub(crate) mod phantom {
    use std::{array::from_fn, collections::VecDeque};

    use ax_ecc_execution::curves::{bls12_381::Bls12_381, bn254::Bn254};
    use axvm_ecc::{algebra::field::FieldExtension, halo2curves::ff, pairing::FinalExp};
    use axvm_ecc_constants::{BLS12381, BN254};
    use axvm_instructions::PhantomDiscriminant;
    use eyre::bail;
    use p3_field::PrimeField32;

    use crate::{
        arch::{PairingCurve, PhantomSubExecutor, Streams},
        rv32im::adapters::unsafe_read_rv32_register,
        system::memory::MemoryController,
    };

    pub struct PairingHintSubEx;

    impl<F: PrimeField32> PhantomSubExecutor<F> for PairingHintSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            _: u16,
        ) -> eyre::Result<()> {
            let rs = unsafe_read_rv32_register(memory, a);
            let b = b.as_canonical_u32();
            hint_final_exp(memory, &mut streams.hint_stream, rs, b)
        }
    }

    /// Return success as bool
    // TODO: return descriptive error type instead of eyre::Result
    fn hint_final_exp<F: PrimeField32>(
        memory: &MemoryController<F>,
        hint_stream: &mut VecDeque<F>,
        mut rs: u32,
        b: u32,
    ) -> eyre::Result<()> {
        match PairingCurve::from_repr(b as usize) {
            Some(PairingCurve::Bn254) => {
                use axvm_ecc::halo2curves::bn256::{Fq, Fq12, Fq2};
                const N: usize = 32;
                debug_assert_eq!(BN254.NUM_LIMBS, N); // TODO: make this const instead of static
                let f: Fq12 = Fq12::from_coeffs(from_fn(|_| {
                    Fq2::from_coeffs(from_fn(|_| {
                        let fp = read_fp::<N, F, Fq>(memory, rs).unwrap(); // TODO: better error handling
                        rs += N as u32;
                        fp
                    }))
                }));
                let (c, u) = Bn254::final_exp_hint(&f);
                hint_stream.clear();
                hint_stream.extend(
                    c.to_coeffs()
                        .into_iter()
                        .chain(u.to_coeffs())
                        .flat_map(|fp2| fp2.to_coeffs())
                        .flat_map(|fp| fp.to_bytes())
                        .map(F::from_canonical_u8),
                );
            }
            Some(PairingCurve::Bls12_381) => {
                use axvm_ecc::halo2curves::bls12_381::{Fq, Fq12, Fq2};
                const N: usize = 48;
                debug_assert_eq!(BLS12381.NUM_LIMBS, N); // TODO: make this const instead of static
                let f: Fq12 = Fq12::from_coeffs(from_fn(|_| {
                    Fq2::from_coeffs(from_fn(|_| {
                        let fp = read_fp::<N, F, Fq>(memory, rs).unwrap();
                        rs += N as u32;
                        fp
                    }))
                }));
                let (c, u) = Bls12_381::final_exp_hint(&f);
                hint_stream.clear();
                hint_stream.extend(
                    c.to_coeffs()
                        .into_iter()
                        .chain(u.to_coeffs())
                        .flat_map(|fp2| fp2.to_coeffs())
                        .flat_map(|fp| fp.to_bytes())
                        .map(F::from_canonical_u8),
                );
            }
            _ => {
                bail!("hint_final_exp: invalid operand b={b}");
            }
        }
        Ok(())
    }

    fn read_fp<const N: usize, F: PrimeField32, Fp: ff::PrimeField>(
        memory: &MemoryController<F>,
        rs: u32,
    ) -> eyre::Result<Fp>
    where
        Fp::Repr: From<[u8; N]>,
    {
        let mut repr = [0u8; N];
        for (i, byte) in repr.iter_mut().enumerate() {
            *byte = memory
                .unsafe_read_cell(F::TWO, F::from_canonical_u32(rs + i as u32))
                .as_canonical_u32()
                .try_into()?;
        }
        Fp::from_repr(repr.into())
            .into_option()
            .ok_or(eyre::eyre!("bad repr"))
    }
}
