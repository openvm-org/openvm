use std::any::Any;

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        cuda::postflight::{
            GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
        },
        prepare_gpu_postflight, to_byte_ptr_bits, GenerationError, PostflightTracegen,
        PreflightOutput, VirtualMachine, VmBuilder,
    },
    system::cuda::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};
use openvm_circuit_primitives::range_tuple::RangeTupleCheckerChipGPU;
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_instructions::{program::Program, LocalOpcode};
use openvm_riscv_circuit::{Rv64ImGpuProverExt, Rv64ImPreflightGpuTracegen};
use openvm_stark_backend::prover::{AirProvingContext, ProvingContext};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        PostflightAccessRegistry, PostflightAccessSchedule, PostflightAccessSpan,
    },
    openvm_instructions::riscv::RV64_MEMORY_AS,
};
#[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
use {
    openvm_circuit::arch::{
        rvr::{cuda::PreflightReplayProgram, PreflightExecution},
        MemoryConfig,
    },
    openvm_cuda_common::stream::GpuDeviceCtx,
    openvm_stark_backend::p3_field::PrimeField32,
};

use super::*;

pub struct Int256GpuProverExt;

pub struct Int256PreflightGpuTracegen<'a> {
    program: &'a GpuPostflightProgram,
    transcript: &'a GpuPostflightTranscript,
    replay_plan: &'a GpuPostflightPlan,
    pending_add_sub: bool,
    pending_bitwise: bool,
    pending_less_than: bool,
    pending_branch_equal: bool,
    pending_branch_less_than: bool,
    pending_mul: bool,
    pending_shift_logical: bool,
    pending_shift_arithmetic: bool,
}

impl<'a> Int256PreflightGpuTracegen<'a> {
    fn opcodes<T: LocalOpcode>(opcodes: impl IntoIterator<Item = T>) -> Vec<u32> {
        opcodes
            .into_iter()
            .map(|opcode| opcode.global_opcode().as_usize() as u32)
            .collect()
    }

    #[doc(hidden)]
    pub fn extension_opcodes() -> Vec<u32> {
        Self::opcodes(Rv64BaseAlu256Opcode::iter())
            .into_iter()
            .chain(Self::opcodes(Rv64Shift256Opcode::iter()))
            .chain(Self::opcodes(Rv64LessThan256Opcode::iter()))
            .chain(Self::opcodes(Rv64BranchEqual256Opcode::iter()))
            .chain(Self::opcodes(Rv64BranchLessThan256Opcode::iter()))
            .chain(Self::opcodes(Rv64Mul256Opcode::iter()))
            .collect()
    }

    #[doc(hidden)]
    #[cfg(feature = "rvr")]
    pub fn register_postflight_access_schedules(
        registry: &mut PostflightAccessRegistry,
    ) -> Result<(), GpuPostflightError> {
        let alu_spans = [
            PostflightAccessSpan::read_fixed(RV64_MEMORY_AS, 0, 4),
            PostflightAccessSpan::read_fixed(RV64_MEMORY_AS, 1, 4),
            PostflightAccessSpan::write_fixed_from_replay_values(RV64_MEMORY_AS, 2, 4),
        ];
        let alu_schedule = PostflightAccessSchedule {
            register_operands: &[2, 3, 1],
            zero_operand_mask: (1 << 6) | (1 << 7),
            register_as_operand: 4,
            memory_as_operand: 5,
            spans: &alu_spans,
        };
        for opcode in Self::opcodes(Rv64BaseAlu256Opcode::iter())
            .into_iter()
            .chain(Self::opcodes(Rv64Shift256Opcode::iter()))
            .chain(Self::opcodes(Rv64LessThan256Opcode::iter()))
            .chain(Self::opcodes(Rv64Mul256Opcode::iter()))
        {
            registry.register(opcode, alu_schedule)?;
        }
        let branch_spans = [
            PostflightAccessSpan::read_fixed(RV64_MEMORY_AS, 0, 4),
            PostflightAccessSpan::read_fixed(RV64_MEMORY_AS, 1, 4),
        ];
        for opcode in Self::opcodes(Rv64BranchEqual256Opcode::iter())
            .into_iter()
            .chain(Self::opcodes(Rv64BranchLessThan256Opcode::iter()))
        {
            registry.register_branch_from_replay_value(
                opcode,
                PostflightAccessSchedule {
                    register_operands: &[1, 2],
                    zero_operand_mask: (1 << 6) | (1 << 7),
                    register_as_operand: 4,
                    memory_as_operand: 5,
                    spans: &branch_spans,
                },
                3,
            )?;
        }
        Ok(())
    }

    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn upload_postflight_program<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<PreflightReplayProgram, GpuPostflightError> {
        let mut registry = PostflightAccessRegistry::default();
        Self::register_postflight_access_schedules(&mut registry)?;
        registry
            .validate_no_native_collisions(Rv64ImPreflightGpuTracegen::postflight_opcode_bases())?;
        PreflightReplayProgram::upload_with_postflight_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn postflight<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &PreflightReplayProgram,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        vm.postflight(
            program,
            execution,
            num_insns,
            Rv64ImPreflightGpuTracegen::postflight_opcode_bases(),
        )
    }

    pub fn new(
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
    ) -> Self {
        let has_any = |opcodes: Vec<u32>| {
            opcodes.into_iter().any(|opcode| {
                !replay_plan
                    .opcode_range(openvm_instructions::VmOpcode::from_usize(opcode as usize))
                    .is_empty()
            })
        };
        Self {
            program,
            transcript,
            replay_plan,
            pending_add_sub: has_any(
                Self::opcodes(Rv64BaseAlu256Opcode::iter())
                    .into_iter()
                    .take(2)
                    .collect(),
            ),
            pending_bitwise: has_any(
                Self::opcodes(Rv64BaseAlu256Opcode::iter())
                    .into_iter()
                    .skip(2)
                    .collect(),
            ),
            pending_less_than: has_any(Self::opcodes(Rv64LessThan256Opcode::iter())),
            pending_branch_equal: has_any(Self::opcodes(Rv64BranchEqual256Opcode::iter())),
            pending_branch_less_than: has_any(Self::opcodes(Rv64BranchLessThan256Opcode::iter())),
            pending_mul: has_any(Self::opcodes(Rv64Mul256Opcode::iter())),
            pending_shift_logical: has_any(
                Self::opcodes(Rv64Shift256Opcode::iter())
                    .into_iter()
                    .take(2)
                    .collect(),
            ),
            pending_shift_arithmetic: has_any(
                Self::opcodes(Rv64Shift256Opcode::iter())
                    .into_iter()
                    .skip(2)
                    .collect(),
            ),
        }
    }

    pub fn generate_for_chip(
        &mut self,
        chip: &dyn Any,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuPostflightError> {
        macro_rules! generate {
            ($chip_ty:ty, $pending:ident) => {
                if let Some(chip) = chip.downcast_ref::<$chip_ty>() {
                    self.$pending = false;
                    return chip
                        .generate_proving_ctx_from_postflight(
                            self.program,
                            self.transcript,
                            self.replay_plan,
                        )
                        .map(Some);
                }
            };
        }
        generate!(AddSub256ChipGpu, pending_add_sub);
        generate!(BitwiseLogic256ChipGpu, pending_bitwise);
        generate!(LessThan256ChipGpu, pending_less_than);
        generate!(BranchEqual256ChipGpu, pending_branch_equal);
        generate!(BranchLessThan256ChipGpu, pending_branch_less_than);
        generate!(Multiplication256ChipGpu, pending_mul);
        generate!(ShiftLogical256ChipGpu, pending_shift_logical);
        generate!(ShiftRightArithmetic256ChipGpu, pending_shift_arithmetic);
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuPostflightError> {
        let pending = [
            (self.pending_add_sub, "AddSub256"),
            (self.pending_bitwise, "BitwiseLogic256"),
            (self.pending_less_than, "LessThan256"),
            (self.pending_branch_equal, "BranchEqual256"),
            (self.pending_branch_less_than, "BranchLessThan256"),
            (self.pending_mul, "Multiplication256"),
            (self.pending_shift_logical, "ShiftLogical256"),
            (self.pending_shift_arithmetic, "ShiftRightArithmetic256"),
        ]
        .into_iter()
        .filter_map(|(pending, name)| pending.then_some(name))
        .collect::<Vec<_>>();
        if pending.is_empty() {
            Ok(())
        } else {
            Err(GpuPostflightError::InvalidTranscript(format!(
                "Int256 preflight GPU tracegen did not visit producers {pending:?}"
            )))
        }
    }

    pub fn generate_proving_ctx<VB>(
        self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        let extension_opcodes = Self::extension_opcodes();
        let rv64 = Rv64ImPreflightGpuTracegen::new_after_claiming_extension_opcodes(
            self.program,
            self.transcript,
            self.replay_plan,
            &extension_opcodes,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        vm.generate_preflight_proving_ctx(
            self.program,
            self.transcript,
            self.replay_plan,
            (self, rv64),
            |(tracegen, rv64), chip| {
                if let Some(ctx) = tracegen
                    .generate_for_chip(chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
                {
                    Ok(ctx)
                } else {
                    rv64.generate_for_chip(chip)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
                }
            },
            |(tracegen, rv64)| {
                tracegen
                    .finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
                rv64.finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
            },
        )
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, Int256> for Int256GpuProverExt {
    fn extend_prover(
        &self,
        extension: &Int256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                    range_checker.device_ctx.clone(),
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv64AddSub256Air>()?;
        let add_sub =
            AddSub256ChipGpu::new(range_checker.clone(), byte_ptr_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(add_sub);

        inventory.next_air::<Rv64BitwiseLogic256Air>()?;
        let bitwise = BitwiseLogic256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(bitwise);

        inventory.next_air::<Rv64LessThan256Air>()?;
        let lt =
            LessThan256ChipGpu::new(range_checker.clone(), byte_ptr_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64BranchEqual256Air>()?;
        let beq = BranchEqual256ChipGpu::new(
            range_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThan256Air>()?;
        let blt = BranchLessThan256ChipGpu::new(
            range_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64Multiplication256Air>()?;
        let mult = Multiplication256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64ShiftLogical256Air>()?;
        let shift_logical = ShiftLogical256ChipGpu::new(
            range_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift_logical);

        inventory.next_air::<Rv64ShiftRightArithmetic256Air>()?;
        let shift_right_arithmetic = ShiftRightArithmetic256ChipGpu::new(
            range_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift_right_arithmetic);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Int256Rv64GpuBuilder;

impl PostflightTracegen<GpuBabyBearPoseidon2Engine> for Int256Rv64GpuBuilder {
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
        Int256PreflightGpuTracegen::new(program, &transcript, &replay_plan).generate_proving_ctx(vm)
    }
}

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Int256Rv64GpuBuilder {
    type VmConfig = Int256Rv64Config;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &Int256Rv64Config,
        circuit: AirInventory<<E as StarkEngine>::SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<<E as StarkEngine>::SC, <E as StarkEngine>::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(&Rv64ImGpuProverExt, &config.rv64i, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&Rv64ImGpuProverExt, &config.rv64m, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&Rv64ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&Int256GpuProverExt, &config.bigint, inventory)?;
        Ok(chip_complex)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests;
