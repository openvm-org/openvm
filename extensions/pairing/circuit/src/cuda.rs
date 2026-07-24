//! GPU builder where the [Rv64ModularBuilder], [AlgebraProverExt], and [EccProverExt] will use
//! either cuda tracegen or hybrid CPU tracegen depending on what [openvm_algebra_circuit] and
//! [openvm_ecc_circuit] crates export.
use openvm_algebra_circuit::{AlgebraProverExt, Rv64ModularBuilder};
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex,
        VmProverExtension,
    },
    system::cuda::SystemChipInventoryGPU,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_ecc_circuit::EccProverExt;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::{
        rvr::cuda::{GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript},
        GenerationError, VirtualMachine,
    },
    openvm_ecc_circuit::WeierstrassRvrGpuTracegen,
    openvm_stark_backend::prover::ProvingContext,
};

use crate::{PairingProverExt, Rv64PairingConfig};

#[derive(Clone)]
pub struct Rv64PairingGpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

#[cfg(feature = "rvr")]
impl Rv64PairingGpuBuilder {
    /// Runs the concrete system/RV64 + modular/Fp2 + Weierstrass inventory walk.
    /// Pairing hints use the existing system PHANTOM producer and add no AIR.
    pub fn generate_proving_ctx_from_rvr(
        vm: &mut VirtualMachine<E, Self>,
        config: &Rv64PairingConfig,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        WeierstrassRvrGpuTracegen::new(&config.weierstrass, program, transcript, replay_plan)
            .generate_proving_ctx(vm, &config.modular.modular, Some(&config.fp2))
    }
}

impl VmBuilder<E> for Rv64PairingGpuBuilder {
    type VmConfig = Rv64PairingConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64PairingConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ModularBuilder,
            &config.modular,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&AlgebraProverExt, &config.fp2, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&EccProverExt, &config.weierstrass, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&PairingProverExt, &config.pairing, inventory)?;
        Ok(chip_complex)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests {
    use openvm_circuit::{
        arch::{
            rvr::{cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightTranscript},
            VirtualMachine, VmExecutor,
        },
        utils::{test_gpu_engine, test_system_config},
    };
    use openvm_instructions::{
        exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
    };
    use openvm_pairing_guest::bn254::BN254_COMPLEX_STRUCT_NAME;
    use openvm_stark_backend::StarkEngine;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rvr_state::PreflightProgramEvent;

    use super::*;
    use crate::PairingCurve;

    #[test]
    fn pairing_config_record_free_inventory_proves() {
        let program = Program::from_instructions(&[Instruction::<BabyBear>::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [0; 5],
        )]);
        let transcript = RvrPreflightTranscript {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let mut config = Rv64PairingConfig::new(
            vec![PairingCurve::Bn254],
            vec![BN254_COMPLEX_STRUCT_NAME.to_string()],
        );
        *config.as_mut() = test_system_config();
        let exe = VmExe::new(program.clone());
        let executor = VmExecutor::new(config.clone()).unwrap();
        let state = executor
            .interpreter_instance(&exe)
            .unwrap()
            .create_initial_vm_state(Vec::<Vec<u8>>::new());
        let (mut vm, pk) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            Rv64PairingGpuBuilder,
            config.clone(),
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let gpu_program = GpuRvrProgram::upload(
            &program,
            &config.modular.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_transcript(&transcript, RvrPreflightEndpoint::Terminated)
            .unwrap();
        let proving_ctx = Rv64PairingGpuBuilder::generate_proving_ctx_from_rvr(
            &mut vm,
            &config,
            &gpu_program,
            &gpu_transcript,
            &replay_plan,
        )
        .unwrap();
        drop(replay_plan);
        drop(gpu_transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();
    }
}
