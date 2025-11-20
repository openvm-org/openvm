use derive_more::derive::From;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Zero};
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
        VmProverExtension,
    },
    system::{memory::online::GuestMemory, phantom::PhantomExecutor},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_ecc_circuit::CurveConfig;
use openvm_instructions::{riscv::RV32_MEMORY_AS, PhantomDiscriminant};
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

use crate::{bls12_381::arkworks as bls12_ark, bn254::arkworks as bn254_ark};

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
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
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
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_instructions::{
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
        PhantomDiscriminant,
    };
    use openvm_pairing_guest::{bls12_381::BLS12_381_NUM_LIMBS, bn254::BN254_NUM_LIMBS};
    use openvm_rv32im_circuit::adapters::{memory_read, read_rv32_register};
    use openvm_stark_backend::p3_field::Field;
    use rand::rngs::StdRng;

    use super::*;

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
                if p_len != q_len {
                    bail!("hint_pairing: p_len={p_len} != q_len={q_len}");
                }

                let raw_p = read_g1::<BN254_NUM_LIMBS>(memory, p_ptr, p_len);
                let raw_q = read_g2::<BN254_NUM_LIMBS>(memory, q_ptr, q_len);

                let p = bn254_ark::parse_g1_points(raw_p);
                let q = bn254_ark::parse_g2_points(raw_q);
                let bytes = bn254_ark::pairing_hint_bytes(&p, &q);

                hint_stream.clear();
                hint_stream.extend(bytes.into_iter().map(F::from_canonical_u8));
            }
            Some(PairingCurve::Bls12_381) => {
                if p_len != q_len {
                    bail!("hint_pairing: p_len={p_len} != q_len={q_len}");
                }

                let raw_p = read_g1::<BLS12_381_NUM_LIMBS>(memory, p_ptr, p_len);
                let raw_q = read_g2::<BLS12_381_NUM_LIMBS>(memory, q_ptr, q_len);

                let p = bls12_ark::parse_g1_points(raw_p);
                let q = bls12_ark::parse_g2_points(raw_q);
                let ark_bytes = bls12_ark::pairing_hint_bytes(&p, &q);

                hint_stream.clear();
                hint_stream.extend(ark_bytes.into_iter().map(F::from_canonical_u8));
            }
            _ => {
                bail!("hint_pairing: invalid PairingCurve={c_upper}");
            }
        }
        Ok(())
    }
}

fn read_g1<const N: usize>(memory: &GuestMemory, base_ptr: u32, len: u32) -> Vec<[[u8; N]; 2]> {
    (0..len)
        .map(|i| {
            let ptr = base_ptr + i * 2 * (N as u32);
            let x = unsafe { memory.read::<u8, N>(RV32_MEMORY_AS, ptr) };
            let y = unsafe { memory.read::<u8, N>(RV32_MEMORY_AS, ptr + N as u32) };
            [x, y]
        })
        .collect()
}

fn read_g2<const N: usize>(
    memory: &GuestMemory,
    base_ptr: u32,
    len: u32,
) -> Vec<[[[u8; N]; 2]; 2]> {
    (0..len)
        .map(|i| {
            let offset = i * 4 * (N as u32);
            let x_c0 = unsafe { memory.read::<u8, N>(RV32_MEMORY_AS, base_ptr + offset) };
            let x_c1 =
                unsafe { memory.read::<u8, N>(RV32_MEMORY_AS, base_ptr + offset + N as u32) };
            let y_c0 =
                unsafe { memory.read::<u8, N>(RV32_MEMORY_AS, base_ptr + offset + 2 * N as u32) };
            let y_c1 =
                unsafe { memory.read::<u8, N>(RV32_MEMORY_AS, base_ptr + offset + 3 * N as u32) };
            [[x_c0, x_c1], [y_c0, y_c1]]
        })
        .collect()
}
