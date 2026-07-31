//! GPU builder where the [Rv64ModularBuilder], [AlgebraProverExt], and [EccProverExt] will use
//! either cuda tracegen or hybrid CPU tracegen depending on what [openvm_algebra_circuit] and
//! [openvm_ecc_circuit] crates export.
use openvm_algebra_circuit::{AlgebraProverExt, Rv64ModularBuilder};
use openvm_circuit::{
    arch::{
        cuda::postflight::{GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript},
        prepare_gpu_postflight, AirInventory, ChipInventoryError, GenerationError,
        PostflightTracegen, PreflightOutput, VirtualMachine, VmBuilder, VmChipComplex,
        VmProverExtension,
    },
    system::cuda::SystemChipInventoryGPU,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_ecc_circuit::{EccProverExt, WeierstrassPreflightGpuTracegen};
use openvm_instructions::program::Program;
use openvm_stark_backend::prover::ProvingContext;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};

use crate::{PairingProverExt, Rv64PairingConfig};

#[derive(Clone)]
pub struct Rv64PairingGpuBuilder;

impl PostflightTracegen<GpuBabyBearPoseidon2Engine> for Rv64PairingGpuBuilder {
    type Prepared = GpuPostflightProgram;

    fn prepare_postflight(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
        program: &Program<BabyBear>,
    ) -> Result<Self::Prepared, GenerationError> {
        prepare_gpu_postflight(vm, program)
    }

    fn generate_proving_ctx(
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
        _host_program: &Program<BabyBear>,
        program: &Self::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let (transcript, replay_plan) = vm
            .postflight_history(program, output)
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let config = vm.config().clone();
        Self::generate_proving_ctx_from_postflight(vm, &config, program, &transcript, &replay_plan)
    }
}

type E = GpuBabyBearPoseidon2Engine;

impl Rv64PairingGpuBuilder {
    /// Runs the concrete system/RV64 + modular/Fp2 + Weierstrass inventory walk.
    /// Pairing hints use the existing system PHANTOM producer and add no AIR.
    pub fn generate_proving_ctx_from_postflight(
        vm: &mut VirtualMachine<E, Self>,
        config: &Rv64PairingConfig,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        WeierstrassPreflightGpuTracegen::new(&config.weierstrass, program, transcript, replay_plan)
            .generate_proving_ctx(vm, &config.modular.modular, Some(&config.fp2))
    }
}

impl VmBuilder<E> for Rv64PairingGpuBuilder {
    type VmConfig = Rv64PairingConfig;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &Rv64PairingConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<BabyBearPoseidon2Config, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ModularBuilder,
            &config.modular,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(&AlgebraProverExt, &config.fp2, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&EccProverExt, &config.weierstrass, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&PairingProverExt, &config.pairing, inventory)?;
        Ok(chip_complex)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests {
    use halo2curves_axiom::{
        bls12_381::{G1Affine as Bls12_381G1Affine, G2Affine as Bls12_381G2Affine},
        bn256::{G1Affine as Bn254G1Affine, G2Affine as Bn254G2Affine},
    };
    use openvm_circuit::{
        arch::{
            cuda::postflight::GpuPostflightProgram,
            rvr::{cuda::CheckpointReplayProgram, PreflightEndpoint, PreflightLimits},
            PreflightHistory, PreflightMemoryLog, VirtualMachine, VmExecutor,
        },
        utils::{test_gpu_engine, test_system_config},
    };
    use openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        instruction::Instruction,
        program::Program,
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        LocalOpcode, PhantomDiscriminant, SystemOpcode,
    };
    use openvm_pairing_guest::{
        bls12_381::BLS12_381_COMPLEX_STRUCT_NAME, bn254::BN254_COMPLEX_STRUCT_NAME,
    };
    use openvm_pairing_transpiler::PairingPhantom;
    use openvm_riscv_circuit::Rv64ImPreflightGpuTracegen;
    use openvm_riscv_transpiler::{Rv64HintStoreOpcode, Rv64JalLuiOpcode};
    use openvm_stark_backend::{p3_field::PrimeCharacteristicRing, StarkEngine};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rvr_state::PreflightProgramEvent;

    use super::*;
    use crate::PairingCurve;

    fn reg(index: usize) -> usize {
        index * RV64_REGISTER_NUM_LIMBS
    }

    fn insert_bytes(
        memory: &mut SparseMemoryImage,
        address_space: u32,
        pointer: u32,
        bytes: impl IntoIterator<Item = u8>,
    ) {
        memory.extend(
            bytes
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((address_space, pointer + offset as u32), byte)),
        );
    }

    fn pairing_inputs(curve: PairingCurve) -> (Vec<u8>, Vec<u8>) {
        match curve {
            PairingCurve::Bn254 => {
                let p = Bn254G1Affine::generator();
                let q = Bn254G2Affine::generator();
                let p = [p.x.to_bytes(), p.y.to_bytes()].concat();
                let q = [
                    q.x.c0.to_bytes(),
                    q.x.c1.to_bytes(),
                    q.y.c0.to_bytes(),
                    q.y.c1.to_bytes(),
                ]
                .concat();
                (p, q)
            }
            PairingCurve::Bls12_381 => {
                let p = Bls12_381G1Affine::generator();
                let q = Bls12_381G2Affine::generator();
                let p = [p.x.to_bytes(), p.y.to_bytes()].concat();
                let q = [
                    q.x.c0.to_bytes(),
                    q.x.c1.to_bytes(),
                    q.y.c0.to_bytes(),
                    q.y.c1.to_bytes(),
                ]
                .concat();
                (p, q)
            }
        }
    }

    fn pairing_complex_name(curve: PairingCurve) -> &'static str {
        match curve {
            PairingCurve::Bn254 => BN254_COMPLEX_STRUCT_NAME,
            PairingCurve::Bls12_381 => BLS12_381_COMPLEX_STRUCT_NAME,
        }
    }

    fn prove_pairing_hint_checkpoint_boundary(curve: PairingCurve) {
        const P_HEADER: u32 = 0x100;
        const Q_HEADER: u32 = 0x110;
        const P_DATA: u32 = 0x200;
        const Q_DATA: u32 = 0x400;
        const HINT_DESTINATION: u32 = 0x800;

        let instructions = [
            Instruction::phantom(
                PhantomDiscriminant(PairingPhantom::HintFinalExp as u16),
                BabyBear::from_usize(reg(1)),
                BabyBear::from_usize(reg(2)),
                curve as u16,
            ),
            Instruction::from_usize(
                Rv64JalLuiOpcode::JAL.global_opcode(),
                [0, 0, 4, RV64_REGISTER_AS as usize, 0, 0, 0],
            ),
            Instruction::from_usize(
                Rv64HintStoreOpcode::HINT_STORED.global_opcode(),
                [
                    0,
                    reg(3),
                    0,
                    RV64_REGISTER_AS as usize,
                    RV64_MEMORY_AS as usize,
                ],
            ),
            Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 7]),
        ];
        let program = Program::from_instructions(&instructions);
        let (p, q) = pairing_inputs(curve);
        let mut initial_memory = SparseMemoryImage::default();
        for (register, value) in [(1, P_HEADER), (2, Q_HEADER), (3, HINT_DESTINATION)] {
            insert_bytes(
                &mut initial_memory,
                RV64_REGISTER_AS,
                reg(register) as u32,
                u64::from(value).to_le_bytes(),
            );
        }
        insert_bytes(
            &mut initial_memory,
            RV64_MEMORY_AS,
            P_HEADER,
            u64::from(P_DATA)
                .to_le_bytes()
                .into_iter()
                .chain(1u64.to_le_bytes()),
        );
        insert_bytes(
            &mut initial_memory,
            RV64_MEMORY_AS,
            Q_HEADER,
            u64::from(Q_DATA)
                .to_le_bytes()
                .into_iter()
                .chain(1u64.to_le_bytes()),
        );
        insert_bytes(&mut initial_memory, RV64_MEMORY_AS, P_DATA, p);
        insert_bytes(&mut initial_memory, RV64_MEMORY_AS, Q_DATA, q);

        let exe = VmExe::new(program.clone()).with_init_memory(initial_memory);
        let mut config =
            Rv64PairingConfig::new(vec![curve], vec![pairing_complex_name(curve).to_string()]);
        *config.as_mut() = test_system_config();
        let executor = VmExecutor::new(config.clone()).unwrap();
        let checkpoint = executor.preflight_instance(&exe).unwrap();
        let state = checkpoint.create_initial_vm_state(Vec::<Vec<u8>>::new());
        let (mut vm, pk) = VirtualMachine::new_with_keygen(
            test_gpu_engine(),
            Rv64PairingGpuBuilder,
            config.clone(),
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let gpu_program = CheckpointReplayProgram::upload(
            &program,
            &config.modular.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();

        let first = checkpoint
            .execute_from_state_for(state, PreflightLimits::new(2, 0, 1))
            .unwrap();
        assert_eq!(first.endpoint, PreflightEndpoint::Suspended);
        assert!(first.state.streams.hint_stream.remaining() > 8);
        let hint_bytes = first.state.streams.hint_stream.remaining();
        let (transcript, replay_plan) = vm
            .postflight(
                &gpu_program,
                &first,
                first.retired,
                Rv64ImPreflightGpuTracegen::postflight_opcode_bases(),
            )
            .unwrap();
        let proving_ctx = Rv64PairingGpuBuilder::generate_proving_ctx_from_postflight(
            &mut vm,
            &config,
            gpu_program.program(),
            &transcript,
            &replay_plan,
        )
        .unwrap();
        drop(replay_plan);
        drop(transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();

        vm.transport_init_memory_to_device(&first.state.memory);
        let second = checkpoint
            .execute_from_state_for(first.state, PreflightLimits::new(2, 1, 1))
            .unwrap();
        assert_eq!(second.endpoint, PreflightEndpoint::Terminated);
        assert_eq!(second.transcript.replay_values.len(), 1);
        assert_eq!(
            second.state.streams.hint_stream.remaining(),
            hint_bytes - u64::BITS as usize / 8
        );
        let (transcript, replay_plan) = vm
            .postflight(
                &gpu_program,
                &second,
                second.retired,
                Rv64ImPreflightGpuTracegen::postflight_opcode_bases(),
            )
            .unwrap();
        let proving_ctx = Rv64PairingGpuBuilder::generate_proving_ctx_from_postflight(
            &mut vm,
            &config,
            gpu_program.program(),
            &transcript,
            &replay_plan,
        )
        .unwrap();
        drop(replay_plan);
        drop(transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();
    }

    #[test]
    fn bn254_pairing_hint_crosses_checkpoint_boundary_without_records() {
        prove_pairing_hint_checkpoint_boundary(PairingCurve::Bn254);
    }

    #[test]
    fn bls12_381_pairing_hint_crosses_checkpoint_boundary_without_records() {
        prove_pairing_hint_checkpoint_boundary(PairingCurve::Bls12_381);
    }

    #[test]
    fn pairing_config_record_free_inventory_proves() {
        let program = Program::from_instructions(&[Instruction::<BabyBear>::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [0; 5],
        )]);
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
            ],
            memory: PreflightMemoryLog::default(),
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
        let gpu_program = GpuPostflightProgram::upload(
            &program,
            &config.modular.system.memory_config,
            &vm.engine.device().device_ctx,
        )
        .unwrap();
        let (gpu_transcript, replay_plan) = gpu_program
            .upload_history_for_test(&program, &history, Some(0))
            .unwrap();
        let proving_ctx = Rv64PairingGpuBuilder::generate_proving_ctx_from_postflight(
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
