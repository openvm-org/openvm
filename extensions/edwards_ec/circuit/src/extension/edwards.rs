use std::sync::Arc;

use num_bigint::BigUint;
use once_cell::sync::Lazy;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerBus,
};
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_te_guest::{
    algebra::IntMod,
    ed25519::{CURVE_A as ED25519_A, CURVE_D as ED25519_D, ED25519_MODULUS, ED25519_ORDER},
};
use openvm_te_transpiler::Rv32EdwardsOpcode;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use strum::EnumCount;

use crate::{
    get_te_add_air, get_te_add_chip, get_te_add_step, EdwardsAir, EdwardsCpuProverExt,
    TeAddExecutor,
};

#[serde_as]
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct CurveConfig {
    /// The name of the curve struct as defined by moduli_declare.
    pub struct_name: String,
    /// The coordinate modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub modulus: BigUint,
    /// The scalar field modulus of the curve.
    #[serde_as(as = "DisplayFromStr")]
    pub scalar: BigUint,
    /// The coefficient a of ax^2 + y^2 = 1 + d x^2 y^2.
    #[serde_as(as = "DisplayFromStr")]
    pub a: BigUint,
    /// The coefficient d of ax^2 + y^2 = 1 + d x^2 y^2.
    #[serde_as(as = "DisplayFromStr")]
    pub d: BigUint,
}

pub static ED25519_CONFIG: Lazy<CurveConfig> = Lazy::new(|| CurveConfig {
    struct_name: "Ed25519Point".to_string(),
    modulus: ED25519_MODULUS.clone(),
    scalar: ED25519_ORDER.clone(),
    a: BigUint::from_bytes_le(ED25519_A.as_le_bytes()),
    d: BigUint::from_bytes_le(ED25519_D.as_le_bytes()),
});

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct EdwardsExtension {
    pub supported_curves: Vec<CurveConfig>,
}

impl EdwardsExtension {
    pub fn generate_te_init(&self) -> String {
        let supported_curves = self
            .supported_curves
            .iter()
            .map(|curve_config| format!("\"{}\"", curve_config.struct_name))
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_te_guest::te_macros::te_init! {{ {supported_curves} }}")
    }
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum EdwardsExtensionExecutor {
    // 32 limbs prime for ed25519
    TeAddRv32_32(TeAddExecutor<2, 32>),
}

impl<F: PrimeField32> VmExecutionExtension<F> for EdwardsExtension {
    type Executor = EdwardsExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, EdwardsExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        // TODO: somehow get the range checker bus from `ExecutorInventory`
        let dummy_range_checker_bus = VariableRangeCheckerBus::new(u16::MAX, 16);
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset = Rv32EdwardsOpcode::CLASS_OFFSET + i * Rv32EdwardsOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };
                let add = get_te_add_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                    curve.d.clone(),
                );

                inventory.add_executor(
                    EdwardsExtensionExecutor::TeAddRv32_32(add),
                    ((Rv32EdwardsOpcode::TE_ADD as usize)
                        ..=(Rv32EdwardsOpcode::SETUP_TE_ADD as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for EdwardsExtension {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker_bus = inventory.range_checker().bus;
        let pointer_max_bits = inventory.pointer_max_bits();

        let bitwise_lu = {
            // A trick to get around Rust's borrow rules
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset = Rv32EdwardsOpcode::CLASS_OFFSET + i * Rv32EdwardsOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                let add = get_te_add_air::<2, 32>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                    curve.d.clone(),
                );
                inventory.add_air(add);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, EdwardsExtension> for EdwardsCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        extension: &EdwardsExtension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };
        for curve in extension.supported_curves.iter() {
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<EdwardsAir<2, 2, 32>>()?;
                let add = get_te_add_chip::<Val<SC>, 2, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                    curve.d.clone(),
                );
                inventory.add_executor_chip(add);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}
