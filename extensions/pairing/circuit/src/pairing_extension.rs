use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use ax_mod_circuit_builder::ExprBuilderConfig;
use axvm_circuit::{
    arch::{VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError},
    rv32im::adapters::{Rv32VecHeapAdapterChip, Rv32VecHeapTwoReadsAdapterChip},
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_ecc_circuit::CurveConfig;
use axvm_ecc_constants::{BLS12381, BN254};
use axvm_instructions::{PairingOpcode, UsizeOpcode};
use derive_more::derive::From;
use num_bigint_dig::BigUint;
use num_traits::Zero;
use p3_field::PrimeField32;
use strum::EnumCount;

use super::*;

// All the supported pairing curves.
#[derive(Clone, Debug)]
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
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let class_offset = PairingOpcode::default_offset() + i * PairingOpcode::COUNT;
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleStepRv32_32(miller_double),
                        [class_offset + PairingOpcode::MILLER_DOUBLE_STEP as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleAndAddStepRv32_32(
                            miller_double_and_add,
                        ),
                        [class_offset + PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EvaluateLineRv32_32(eval_line),
                        [class_offset + PairingOpcode::EVALUATE_LINE as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMul013By013(mul013),
                        [class_offset + PairingOpcode::MUL_BY_013 as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMulBy01234(mul01234),
                        [class_offset + PairingOpcode::MUL_BY_01234 as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleStepRv32_48(miller_double),
                        [class_offset + PairingOpcode::MILLER_DOUBLE_STEP as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::MillerDoubleAndAddStepRv32_48(
                            miller_double_and_add,
                        ),
                        [class_offset + PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EvaluateLineRv32_48(eval_line),
                        [class_offset + PairingOpcode::EVALUATE_LINE as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMul023By023(mul023),
                        [class_offset + PairingOpcode::MUL_BY_023 as usize],
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
                        class_offset,
                    );
                    inventory.add_executor(
                        PairingExtensionExecutor::EcLineMulBy02345(mul02345),
                        [class_offset + PairingOpcode::MUL_BY_02345 as usize],
                    )?;
                }
            }
        }

        Ok(inventory)
    }
}
