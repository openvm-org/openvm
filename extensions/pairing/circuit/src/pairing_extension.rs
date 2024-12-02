use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use ax_mod_circuit_builder::ExprBuilderConfig;
use ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::{
    arch::{VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_ecc_circuit::CurveConfig;
use axvm_ecc_constants::{BLS12381, BN254};
use axvm_instructions::{
    Fp12Opcode, PairingOpcode, PairingPhantom, PhantomDiscriminant, UsizeOpcode,
};
use axvm_rv32_adapters::{Rv32VecHeapAdapterChip, Rv32VecHeapTwoReadsAdapterChip};
use derive_more::derive::From;
use num_bigint_dig::BigUint;
use num_traits::Zero;
use strum::{EnumCount, FromRepr};

use super::*;

// All the supported pairing curves.
#[derive(Clone, Copy, Debug, FromRepr)]
#[repr(usize)]
pub enum PairingCurve {
    Bn254,
    Bls12_381,
}

impl PairingCurve {
    pub fn curve_config(&self) -> CurveConfig {
        match self {
            PairingCurve::Bn254 => {
                CurveConfig::new(BN254.MODULUS.clone(), BN254.ORDER.clone(), BigUint::zero())
            }
            PairingCurve::Bls12_381 => CurveConfig::new(
                BLS12381.MODULUS.clone(),
                BLS12381.ORDER.clone(),
                BigUint::zero(),
            ),
        }
    }

    pub fn xi(&self) -> [isize; 2] {
        match self {
            PairingCurve::Bn254 => BN254.XI,
            PairingCurve::Bls12_381 => BLS12381.XI,
        }
    }
}

#[derive(Clone, Debug, derive_new::new)]
pub struct PairingExtension {
    pub supported_curves: Vec<PairingCurve>,
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, AnyEnum)]
pub enum PairingExtensionExecutor<F: PrimeField32> {
    // bn254 (32 limbs)
    MillerDoubleStepRv32_32(MillerDoubleStepChip<F, 4, 8, 32>),
    MillerDoubleAndAddStepRv32_32(MillerDoubleAndAddStepChip<F, 4, 12, 32>),
    EvaluateLineRv32_32(EvaluateLineChip<F, 4, 2, 4, 32>),
    Fp12MulRv32_32(Fp12MulChip<F, 12, 32>),
    EcLineMul013By013(EcLineMul013By013Chip<F, 4, 10, 32>),
    EcLineMulBy01234(EcLineMulBy01234Chip<F, 12, 10, 12, 32>),
    // bls12-381 (48 limbs)
    MillerDoubleStepRv32_48(MillerDoubleStepChip<F, 12, 24, 16>),
    MillerDoubleAndAddStepRv32_48(MillerDoubleAndAddStepChip<F, 12, 36, 16>),
    EvaluateLineRv32_48(EvaluateLineChip<F, 12, 6, 12, 16>),
    Fp12MulRv32_48(Fp12MulChip<F, 36, 16>),
    EcLineMul023By023(EcLineMul023By023Chip<F, 12, 30, 16>),
    EcLineMulBy02345(EcLineMulBy02345Chip<F, 36, 30, 36, 16>),
}

#[derive(ChipUsageGetter, Chip, AnyEnum, From)]
pub enum PairingExtensionPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for PairingExtension {
    type Executor = PairingExtensionExecutor<F>;
    type Periphery = PairingExtensionPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            Arc::clone(chip)
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lu_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };
        for curve in self.supported_curves.iter() {
            let pairing_idx = *curve as usize;
            let pairing_class_offset =
                PairingOpcode::default_offset() + pairing_idx * PairingOpcode::COUNT;
            let fp12_class_offset = Fp12Opcode::default_offset() + pairing_idx * Fp12Opcode::COUNT;
            match curve {
                PairingCurve::Bn254 => {
                    let bn_config = ExprBuilderConfig {
                        modulus: curve.curve_config().modulus.clone(),
                        num_limbs: 32,
                        limb_bits: 8,
                    };
                    let miller_double = MillerDoubleStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 1, 4, 8, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bn_config.clone(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleStepRv32_32(miller_double),
                        [pairing_class_offset + PairingOpcode::MILLER_DOUBLE_STEP as usize],
                    )?;
                    let miller_double_and_add = MillerDoubleAndAddStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 4, 12, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bn_config.clone(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleAndAddStepRv32_32(
                            miller_double_and_add,
                        ),
                        [
                            pairing_class_offset
                                + PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize,
                        ],
                    )?;
                    let eval_line = EvaluateLineChip::new(
                        Rv32VecHeapTwoReadsAdapterChip::<F, 4, 2, 4, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bn_config.clone(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EvaluateLineRv32_32(eval_line),
                        [pairing_class_offset + PairingOpcode::EVALUATE_LINE as usize],
                    )?;
                    let mul013 = EcLineMul013By013Chip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 4, 10, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bn_config.clone(),
                        curve.xi(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMul013By013(mul013),
                        [pairing_class_offset + PairingOpcode::MUL_013_BY_013 as usize],
                    )?;
                    let mul01234 = EcLineMulBy01234Chip::new(
                        Rv32VecHeapTwoReadsAdapterChip::<F, 12, 10, 12, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bn_config.clone(),
                        curve.xi(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMulBy01234(mul01234),
                        [pairing_class_offset + PairingOpcode::MUL_BY_01234 as usize],
                    )?;
                    let fp12_mul = Fp12MulChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 12, 12, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bn_config.clone(),
                        curve.xi(),
                        fp12_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::Fp12MulRv32_32(fp12_mul),
                        [fp12_class_offset + Fp12Opcode::MUL as usize],
                    )?;
                }
                PairingCurve::Bls12_381 => {
                    let bls_config = ExprBuilderConfig {
                        modulus: curve.curve_config().modulus.clone(),
                        num_limbs: 48,
                        limb_bits: 8,
                    };
                    let miller_double = MillerDoubleStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 1, 12, 24, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bls_config.clone(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleStepRv32_48(miller_double),
                        [pairing_class_offset + PairingOpcode::MILLER_DOUBLE_STEP as usize],
                    )?;
                    let miller_double_and_add = MillerDoubleAndAddStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 12, 36, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bls_config.clone(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleAndAddStepRv32_48(
                            miller_double_and_add,
                        ),
                        [
                            pairing_class_offset
                                + PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize,
                        ],
                    )?;
                    let eval_line = EvaluateLineChip::new(
                        Rv32VecHeapTwoReadsAdapterChip::<F, 12, 6, 12, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bls_config.clone(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EvaluateLineRv32_48(eval_line),
                        [pairing_class_offset + PairingOpcode::EVALUATE_LINE as usize],
                    )?;
                    let mul023 = EcLineMul023By023Chip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 12, 30, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bls_config.clone(),
                        curve.xi(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMul023By023(mul023),
                        [pairing_class_offset + PairingOpcode::MUL_023_BY_023 as usize],
                    )?;
                    let mul02345 = EcLineMulBy02345Chip::new(
                        Rv32VecHeapTwoReadsAdapterChip::<F, 36, 30, 36, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bls_config.clone(),
                        curve.xi(),
                        pairing_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMulBy02345(mul02345),
                        [pairing_class_offset + PairingOpcode::MUL_BY_02345 as usize],
                    )?;
                    let fp12_mul = Fp12MulChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 36, 36, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            bitwise_lu_chip.clone(),
                        ),
                        memory_controller.clone(),
                        bls_config.clone(),
                        curve.xi(),
                        fp12_class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::Fp12MulRv32_48(fp12_mul),
                        [fp12_class_offset + Fp12Opcode::MUL as usize],
                    )?;
                }
            }
        }

        builder.add_phantom_sub_executor(
            phantom::PairingHintSubEx,
            PhantomDiscriminant(PairingPhantom::HintFinalExp as u16),
        )?;

        Ok(inventory)
    }
}

pub(crate) mod phantom {
    use std::collections::VecDeque;

    use ax_ecc_execution::curves::{bls12_381::Bls12_381, bn254::Bn254};
    use ax_stark_backend::p3_field::PrimeField32;
    use axvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use axvm_ecc_constants::{BLS12381, BN254};
    use axvm_ecc_guest::{algebra::field::FieldExtension, halo2curves::ff, AffinePoint};
    use axvm_instructions::{
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
        PhantomDiscriminant,
    };
    use axvm_pairing_guest::pairing::{FinalExp, MultiMillerLoop};
    use axvm_rv32im_circuit::adapters::{compose, unsafe_read_rv32_register};
    use eyre::bail;

    use super::PairingCurve;

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
                use axvm_ecc_guest::halo2curves::bn256::{Fq, Fq12, Fq2};
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
                use axvm_ecc_guest::halo2curves::bls12_381::{Fq, Fq12, Fq2};
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
