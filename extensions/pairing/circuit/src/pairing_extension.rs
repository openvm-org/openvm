use derive_more::derive::From;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
        VmProverExtension,
    },
    system::phantom::PhantomExecutor,
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_ecc_circuit::CurveConfig;
use openvm_instructions::PhantomDiscriminant;
use openvm_pairing_guest::{
    bls12_381::{
        BLS12_381_ECC_STRUCT_NAME, BLS12_381_MODULUS, BLS12_381_ORDER, BLS12_381_XI_ISIZE,
    },
    bn254::{BN254_ECC_STRUCT_NAME, BN254_MODULUS, BN254_ORDER, BN254_XI_ISIZE},
};
use openvm_pairing_transpiler::PairingPhantom;
use openvm_stark_backend::{config::StarkGenericConfig, engine::StarkEngine, p3_field::Field};
use serde::{Deserialize, Serialize};
use strum::FromRepr;

// All the supported pairing curves.
#[derive(Clone, Copy, Debug, FromRepr, Serialize, Deserialize)]
#[repr(usize)]
pub enum PairingCurve {
    Bn254,
    Bls12_381,
}

impl PairingCurve {
    pub fn curve_config(&self) -> CurveConfig {
        match self {
            PairingCurve::Bn254 => CurveConfig::new(
                BN254_ECC_STRUCT_NAME.to_string(),
                BN254_MODULUS.clone(),
                BN254_ORDER.clone(),
                BigUint::zero(),
                BigUint::from_u8(3).unwrap(),
            ),
            PairingCurve::Bls12_381 => CurveConfig::new(
                BLS12_381_ECC_STRUCT_NAME.to_string(),
                BLS12_381_MODULUS.clone(),
                BLS12_381_ORDER.clone(),
                BigUint::zero(),
                BigUint::from_u8(4).unwrap(),
            ),
        }
    }

    pub fn xi(&self) -> [isize; 2] {
        match self {
            PairingCurve::Bn254 => BN254_XI_ISIZE,
            PairingCurve::Bls12_381 => BLS12_381_XI_ISIZE,
        }
    }
}

#[derive(Clone, Debug, From, derive_new::new, Serialize, Deserialize)]
pub struct PairingExtension {
    pub supported_curves: Vec<PairingCurve>,
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum PairingExtensionExecutor<F: Field> {
    Phantom(PhantomExecutor<F>),
}

impl<F: Field> VmExecutionExtension<F> for PairingExtension {
    type Executor = PairingExtensionExecutor<F>;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, PairingExtensionExecutor<F>>,
    ) -> Result<(), ExecutorInventoryError> {
        inventory.add_phantom_sub_executor(
            phantom::PairingHintSubEx,
            PhantomDiscriminant(PairingPhantom::HintFinalExp as u16),
        )?;
        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for PairingExtension {
    fn extend_circuit(&self, _inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        Ok(())
    }
}

pub struct PairingProverExt;
impl<E, RA> VmProverExtension<E, RA, PairingExtension> for PairingProverExt
where
    E: StarkEngine,
{
    fn extend_prover(
        &self,
        _: &PairingExtension,
        _inventory: &mut ChipInventory<E::SC, RA, E::PB>,
    ) -> Result<(), ChipInventoryError> {
        Ok(())
    }
}

pub(crate) mod phantom {
    use std::collections::VecDeque;

    use eyre::bail;
    use halo2curves_axiom::ff;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_ecc_guest::{algebra::field::FieldExtension, AffinePoint};
    use openvm_instructions::{
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
        PhantomDiscriminant,
    };
    use openvm_pairing_guest::{
        bls12_381::BLS12_381_NUM_LIMBS,
        bn254::BN254_NUM_LIMBS,
        pairing::{FinalExp, MultiMillerLoop},
    };
    use openvm_rv32im_circuit::adapters::{memory_read, read_rv32_register};
    use openvm_stark_backend::p3_field::Field;
    use rand::rngs::StdRng;

    use super::PairingCurve;

    pub struct PairingHintSubEx;

    impl<F: Field> PhantomSubExecutor<F> for PairingHintSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let rs1 = read_rv32_register(memory, a);
            let rs2 = read_rv32_register(memory, b);
            hint_pairing(memory, &mut streams.hint_stream, rs1, rs2, c_upper)
        }
    }

    fn hint_pairing<F: Field>(
        memory: &GuestMemory,
        hint_stream: &mut VecDeque<F>,
        rs1: u32,
        rs2: u32,
        c_upper: u16,
    ) -> eyre::Result<()> {
        let p_ptr = u32::from_le_bytes(memory_read(memory, RV32_MEMORY_AS, rs1));
        // len in bytes
        let p_len = u32::from_le_bytes(memory_read(
            memory,
            RV32_MEMORY_AS,
            rs1 + RV32_REGISTER_NUM_LIMBS as u32,
        ));

        let q_ptr = u32::from_le_bytes(memory_read(memory, RV32_MEMORY_AS, rs2));
        // len in bytes
        let q_len = u32::from_le_bytes(memory_read(
            memory,
            RV32_MEMORY_AS,
            rs2 + RV32_REGISTER_NUM_LIMBS as u32,
        ));

        match PairingCurve::from_repr(c_upper as usize) {
            Some(PairingCurve::Bn254) => {
                use halo2curves_axiom::bn256::{Fq, Fq12, Fq2};
                use openvm_pairing_guest::halo2curves_shims::bn254::Bn254;
                const N: usize = BN254_NUM_LIMBS;
                if p_len != q_len {
                    bail!("hint_pairing: p_len={p_len} != q_len={q_len}");
                }
                let p = (0..p_len)
                    .map(|i| -> eyre::Result<_> {
                        let ptr = p_ptr + i * 2 * (N as u32);
                        let x = read_fp::<N, Fq>(memory, ptr)?;
                        let y = read_fp::<N, Fq>(memory, ptr + N as u32)?;
                        Ok(AffinePoint::new(x, y))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                let q = (0..q_len)
                    .map(|i| -> eyre::Result<_> {
                        let mut ptr = q_ptr + i * 4 * (N as u32);
                        let mut read_fp2 = || -> eyre::Result<_> {
                            let c0 = read_fp::<N, Fq>(memory, ptr)?;
                            let c1 = read_fp::<N, Fq>(memory, ptr + N as u32)?;
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
                use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
                use openvm_pairing_guest::halo2curves_shims::bls12_381::Bls12_381;
                const N: usize = BLS12_381_NUM_LIMBS;
                if p_len != q_len {
                    bail!("hint_pairing: p_len={p_len} != q_len={q_len}");
                }
                let p = (0..p_len)
                    .map(|i| -> eyre::Result<_> {
                        let ptr = p_ptr + i * 2 * (N as u32);
                        let x = read_fp::<N, Fq>(memory, ptr)?;
                        let y = read_fp::<N, Fq>(memory, ptr + N as u32)?;
                        Ok(AffinePoint::new(x, y))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                let q = (0..q_len)
                    .map(|i| -> eyre::Result<_> {
                        let mut ptr = q_ptr + i * 4 * (N as u32);
                        let mut read_fp2 = || -> eyre::Result<_> {
                            let c0 = read_fp::<N, Fq>(memory, ptr)?;
                            let c1 = read_fp::<N, Fq>(memory, ptr + N as u32)?;
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

    fn read_fp<const N: usize, Fp: ff::PrimeField>(
        memory: &GuestMemory,
        ptr: u32,
    ) -> eyre::Result<Fp>
    where
        Fp::Repr: From<[u8; N]>,
    {
        // SAFETY:
        // - RV32_MEMORY_AS consists of `u8`s
        // - RV32_MEMORY_AS is in bounds
        let repr: &[u8; N] = unsafe {
            memory
                .memory
                .get_slice::<u8>((RV32_MEMORY_AS, ptr), N)
                .try_into()
                .unwrap()
        };
        Fp::from_repr((*repr).into())
            .into_option()
            .ok_or(eyre::eyre!("bad ff::PrimeField repr"))
    }
}
