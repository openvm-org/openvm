pub(crate) mod phantom {
    use std::collections::VecDeque;

    use ax_ecc_execution::curves::{bls12_381::Bls12_381, bn254::Bn254};
    use axvm_ecc::{
        algebra::field::FieldExtension,
        halo2curves::ff,
        pairing::{FinalExp, MultiMillerLoop},
        AffinePoint,
    };
    use axvm_ecc_constants::{BLS12381, BN254};
    use axvm_instructions::{
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
        PhantomDiscriminant,
    };
    use eyre::bail;
    use p3_field::PrimeField32;

    use crate::{
        arch::{PairingCurve, PhantomSubExecutor, Streams},
        rv32im::adapters::{compose, unsafe_read_rv32_register},
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
            c_upper: u16,
        ) -> eyre::Result<()> {
            let rs1 = unsafe_read_rv32_register(memory, a);
            let rs2 = unsafe_read_rv32_register(memory, b);
            hint_pairing(memory, &mut streams.hint_stream, rs1, rs2, c_upper)
        }
    }

    fn hint_pairing<F: PrimeField32>(
        memory: &MemoryController<F>,
        hint_stream: &mut VecDeque<F>,
        rs1: u32,
        rs2: u32,
        c_upper: u16,
    ) -> eyre::Result<()> {
        let p_ptr = compose(memory.unsafe_read(
            F::from_canonical_u32(RV32_MEMORY_AS),
            F::from_canonical_u32(rs1),
        ));
        // len in bytes
        let p_len = compose(memory.unsafe_read(
            F::from_canonical_u32(RV32_MEMORY_AS),
            F::from_canonical_u32(rs1 + RV32_REGISTER_NUM_LIMBS as u32),
        ));
        let q_ptr = compose(memory.unsafe_read(
            F::from_canonical_u32(RV32_MEMORY_AS),
            F::from_canonical_u32(rs2),
        ));
        // len in bytes
        let q_len = compose(memory.unsafe_read(
            F::from_canonical_u32(RV32_MEMORY_AS),
            F::from_canonical_u32(rs2 + RV32_REGISTER_NUM_LIMBS as u32),
        ));

        match PairingCurve::from_repr(c_upper as usize) {
            Some(PairingCurve::Bn254) => {
                use axvm_ecc::halo2curves::bn256::{Fq, Fq12, Fq2};
                const N: usize = 32;
                debug_assert_eq!(BN254.NUM_LIMBS, N); // TODO: make this const instead of static
                if p_len != q_len {
                    bail!("hint_pairing: p_len={p_len} != q_len={q_len}");
                }
                let p = (0..p_len)
                    .map(|i| -> eyre::Result<_> {
                        let ptr = p_ptr + i * 2 * (N as u32);
                        let x = read_fp::<N, F, Fq>(memory, ptr)?;
                        let y = read_fp::<N, F, Fq>(memory, ptr + N as u32)?;
                        Ok(AffinePoint::new(x, y))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                let q = (0..q_len)
                    .map(|i| -> eyre::Result<_> {
                        let mut ptr = q_ptr + i * 4 * (N as u32);
                        let mut read_fp2 = || -> eyre::Result<_> {
                            let c0 = read_fp::<N, F, Fq>(memory, ptr)?;
                            let c1 = read_fp::<N, F, Fq>(memory, ptr + N as u32)?;
                            ptr += 2 * N as u32;
                            Ok(Fq2::new(c0, c1))
                        };
                        let x = read_fp2()?;
                        let y = read_fp2()?;
                        Ok(AffinePoint::new(x, y))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;

                let f: Fq12 = Bn254::multi_miller_loop(&p, &q);
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
                if p_len != q_len {
                    bail!("hint_pairing: p_len={p_len} != q_len={q_len}");
                }
                let p = (0..p_len)
                    .map(|i| -> eyre::Result<_> {
                        let ptr = p_ptr + i * 2 * (N as u32);
                        let x = read_fp::<N, F, Fq>(memory, ptr)?;
                        let y = read_fp::<N, F, Fq>(memory, ptr + N as u32)?;
                        Ok(AffinePoint::new(x, y))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                let q = (0..q_len)
                    .map(|i| -> eyre::Result<_> {
                        let mut ptr = q_ptr + i * 4 * (N as u32);
                        let mut read_fp2 = || -> eyre::Result<_> {
                            let c0 = read_fp::<N, F, Fq>(memory, ptr)?;
                            let c1 = read_fp::<N, F, Fq>(memory, ptr + N as u32)?;
                            ptr += 2 * N as u32;
                            Ok(Fq2 { c0, c1 })
                        };
                        let x = read_fp2()?;
                        let y = read_fp2()?;
                        Ok(AffinePoint::new(x, y))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;

                let f: Fq12 = Bls12_381::multi_miller_loop(&p, &q);
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
                bail!("hint_pairing: invalid PairingCurve={c_upper}");
            }
        }
        Ok(())
    }

    fn read_fp<const N: usize, F: PrimeField32, Fp: ff::PrimeField>(
        memory: &MemoryController<F>,
        ptr: u32,
    ) -> eyre::Result<Fp>
    where
        Fp::Repr: From<[u8; N]>,
    {
        let mut repr = [0u8; N];
        for (i, byte) in repr.iter_mut().enumerate() {
            *byte = memory
                .unsafe_read_cell(
                    F::from_canonical_u32(RV32_MEMORY_AS),
                    F::from_canonical_u32(ptr + i as u32),
                )
                .as_canonical_u32()
                .try_into()?;
        }
        Fp::from_repr(repr.into())
            .into_option()
            .ok_or(eyre::eyre!("bad ff::PrimeField repr"))
    }
}
