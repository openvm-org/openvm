use alu_native_adapter::{AluNativeAdapterAir, AluNativeAdapterExecutor};
use branch_native_adapter::{BranchNativeAdapterAir, BranchNativeAdapterExecutor};
use convert_adapter::{ConvertAdapterAir, ConvertAdapterExecutor};
use derive_more::derive::From;
use loadstore_native_adapter::{NativeLoadStoreAdapterAir, NativeLoadStoreAdapterExecutor};
use native_vectorized_adapter::{NativeVectorizedAdapterAir, NativeVectorizedAdapterExecutor};
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant};
use openvm_native_compiler::{
    CastfOpcode, FieldArithmeticOpcode, FieldExtensionOpcode, FriOpcode, NativeBranchEqualOpcode,
    NativeJalOpcode, NativeLoadStore4Opcode, NativeLoadStoreOpcode, NativePhantom,
    NativeRangeCheckOpcode, Poseidon2Opcode, VerifyBatchOpcode, BLOCK_LOAD_STORE_SIZE,
};
use openvm_poseidon2_air::Poseidon2Config;
use openvm_rv32im_circuit::BranchEqualCoreAir;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{Field, PrimeField32},
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{
    adapters::*,
    branch_eq::{
        NativeBranchEqAir, NativeBranchEqChip, NativeBranchEqExecutor, NativeBranchEqualFiller,
    },
    castf::{CastFAir, CastFChip, CastFCoreAir, CastFCoreFiller, CastFExecutor},
    field_arithmetic::{
        FieldArithmeticAir, FieldArithmeticChip, FieldArithmeticCoreAir, FieldArithmeticCoreFiller,
        FieldArithmeticExecutor,
    },
    field_extension::{
        FieldExtensionAir, FieldExtensionChip, FieldExtensionCoreAir, FieldExtensionCoreFiller,
        FieldExtensionExecutor,
    },
    fri::{
        FriReducedOpeningAir, FriReducedOpeningChip, FriReducedOpeningExecutor,
        FriReducedOpeningFiller,
    },
    jal_rangecheck::{
        JalRangeCheckAir, JalRangeCheckExecutor, JalRangeCheckFiller, NativeJalRangeCheckChip,
    },
    loadstore::{
        NativeLoadStoreAir, NativeLoadStoreChip, NativeLoadStoreCoreAir, NativeLoadStoreCoreFiller,
        NativeLoadStoreExecutor,
    },
    phantom::*,
    poseidon2::{
        air::{NativePoseidon2Air, VerifyBatchBus},
        chip::{NativePoseidon2Executor, NativePoseidon2Filler},
        NativePoseidon2Chip,
    },
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::*;
        pub use self::cuda::{
            NativeGpuProverExt as NativeProverExt,
        };
        pub type NativeBuilder = crate::NativeGpuBuilder;
    } else {
        pub use self::{
            NativeCpuProverExt as NativeProverExt,
        };
        pub type NativeBuilder = crate::NativeCpuBuilder;
    }
}

// ============ VmExtension Implementations ============

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Native;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum NativeExecutor<F: Field> {
    LoadStore(NativeLoadStoreExecutor<1>),
    BlockLoadStore(NativeLoadStoreExecutor<BLOCK_LOAD_STORE_SIZE>),
    BranchEqual(NativeBranchEqExecutor),
    Jal(JalRangeCheckExecutor),
    FieldArithmetic(FieldArithmeticExecutor),
    FieldExtension(FieldExtensionExecutor),
    FriReducedOpening(FriReducedOpeningExecutor),
    VerifyBatch(NativePoseidon2Executor<F, 1>),
}

impl<F: PrimeField32> VmExecutionExtension<F> for Native {
    type Executor = NativeExecutor<F>;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, NativeExecutor<F>>,
    ) -> Result<(), ExecutorInventoryError> {
        let load_store = NativeLoadStoreExecutor::<1>::new(
            NativeLoadStoreAdapterExecutor::new(NativeLoadStoreOpcode::CLASS_OFFSET),
            NativeLoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_store,
            NativeLoadStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let block_load_store = NativeLoadStoreExecutor::<BLOCK_LOAD_STORE_SIZE>::new(
            NativeLoadStoreAdapterExecutor::new(NativeLoadStore4Opcode::CLASS_OFFSET),
            NativeLoadStore4Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            block_load_store,
            NativeLoadStore4Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let branch_equal = NativeBranchEqExecutor::new(
            BranchNativeAdapterExecutor::new(),
            NativeBranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        inventory.add_executor(
            branch_equal,
            NativeBranchEqualOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let jal_rangecheck = JalRangeCheckExecutor;
        inventory.add_executor(
            jal_rangecheck,
            [
                NativeJalOpcode::JAL.global_opcode(),
                NativeRangeCheckOpcode::RANGE_CHECK.global_opcode(),
            ],
        )?;

        let field_arithmetic = FieldArithmeticExecutor::new(AluNativeAdapterExecutor::new());
        inventory.add_executor(
            field_arithmetic,
            FieldArithmeticOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let field_extension = FieldExtensionExecutor::new(NativeVectorizedAdapterExecutor::new());
        inventory.add_executor(
            field_extension,
            FieldExtensionOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let fri_reduced_opening = FriReducedOpeningExecutor::new();
        inventory.add_executor(
            fri_reduced_opening,
            FriOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let verify_batch = NativePoseidon2Executor::<F, 1>::new(Poseidon2Config::default());
        inventory.add_executor(
            verify_batch,
            [
                VerifyBatchOpcode::VERIFY_BATCH.global_opcode(),
                Poseidon2Opcode::PERM_POS2.global_opcode(),
                Poseidon2Opcode::COMP_POS2.global_opcode(),
            ],
        )?;

        inventory.add_phantom_sub_executor(
            NativeHintInputSubEx,
            PhantomDiscriminant(NativePhantom::HintInput as u16),
        )?;

        inventory.add_phantom_sub_executor(
            NativeHintSliceSubEx::<1>,
            PhantomDiscriminant(NativePhantom::HintFelt as u16),
        )?;

        inventory.add_phantom_sub_executor(
            NativeHintBitsSubEx,
            PhantomDiscriminant(NativePhantom::HintBits as u16),
        )?;

        inventory.add_phantom_sub_executor(
            NativePrintSubEx,
            PhantomDiscriminant(NativePhantom::Print as u16),
        )?;

        inventory.add_phantom_sub_executor(
            NativeHintLoadSubEx,
            PhantomDiscriminant(NativePhantom::HintLoad as u16),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Native
where
    Val<SC>: PrimeField32,
{
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;

        let load_store = NativeLoadStoreAir::<1>::new(
            NativeLoadStoreAdapterAir::new(memory_bridge, exec_bridge),
            NativeLoadStoreCoreAir::new(NativeLoadStoreOpcode::CLASS_OFFSET),
        );
        inventory.add_air(load_store);

        let block_load_store = NativeLoadStoreAir::<BLOCK_LOAD_STORE_SIZE>::new(
            NativeLoadStoreAdapterAir::new(memory_bridge, exec_bridge),
            NativeLoadStoreCoreAir::new(NativeLoadStore4Opcode::CLASS_OFFSET),
        );
        inventory.add_air(block_load_store);

        let branch_equal = NativeBranchEqAir::new(
            BranchNativeAdapterAir::new(exec_bridge, memory_bridge),
            BranchEqualCoreAir::new(NativeBranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(branch_equal);

        let jal_rangecheck = JalRangeCheckAir::new(
            ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            range_checker,
        );
        inventory.add_air(jal_rangecheck);

        let field_arithmetic = FieldArithmeticAir::new(
            AluNativeAdapterAir::new(exec_bridge, memory_bridge),
            FieldArithmeticCoreAir::new(),
        );
        inventory.add_air(field_arithmetic);

        let field_extension = FieldExtensionAir::new(
            NativeVectorizedAdapterAir::new(exec_bridge, memory_bridge),
            FieldExtensionCoreAir::new(),
        );
        inventory.add_air(field_extension);

        let fri_reduced_opening = FriReducedOpeningAir::new(
            ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
        );
        inventory.add_air(fri_reduced_opening);

        let verify_batch = NativePoseidon2Air::<_, 1>::new(
            exec_bridge,
            memory_bridge,
            VerifyBatchBus::new(inventory.new_bus_idx()),
            Poseidon2Config::default(),
        );
        inventory.add_air(verify_batch);

        Ok(())
    }
}

pub struct NativeCpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Native> for NativeCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Native,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<NativeLoadStoreAir<1>>()?;
        let load_store = NativeLoadStoreChip::<_, 1>::new(
            NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_store);

        inventory.next_air::<NativeLoadStoreAir<BLOCK_LOAD_STORE_SIZE>>()?;
        let block_load_store = NativeLoadStoreChip::<_, BLOCK_LOAD_STORE_SIZE>::new(
            NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(block_load_store);

        inventory.next_air::<NativeBranchEqAir>()?;
        let branch_eq = NativeBranchEqChip::new(
            NativeBranchEqualFiller::new(BranchNativeAdapterFiller),
            mem_helper.clone(),
        );

        inventory.add_executor_chip(branch_eq);

        inventory.next_air::<JalRangeCheckAir>()?;
        let jal_rangecheck = NativeJalRangeCheckChip::new(
            JalRangeCheckFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jal_rangecheck);

        inventory.next_air::<FieldArithmeticAir>()?;
        let field_arithmetic = FieldArithmeticChip::new(
            FieldArithmeticCoreFiller::new(AluNativeAdapterFiller),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(field_arithmetic);

        inventory.next_air::<FieldExtensionAir>()?;
        let field_extension = FieldExtensionChip::new(
            FieldExtensionCoreFiller::new(NativeVectorizedAdapterFiller),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(field_extension);

        inventory.next_air::<FriReducedOpeningAir>()?;
        let fri_reduced_opening =
            FriReducedOpeningChip::new(FriReducedOpeningFiller::new(), mem_helper.clone());
        inventory.add_executor_chip(fri_reduced_opening);

        inventory.next_air::<NativePoseidon2Air<Val<SC>, 1>>()?;
        let poseidon2 = NativePoseidon2Chip::<_, 1>::new(
            NativePoseidon2Filler::new(Poseidon2Config::default()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(poseidon2);

        Ok(())
    }
}

pub(crate) mod phantom {
    use eyre::bail;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_instructions::PhantomDiscriminant;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};
    use rand::rngs::StdRng;

    pub struct NativeHintInputSubEx;
    pub struct NativeHintSliceSubEx<const N: usize>;
    pub struct NativePrintSubEx;
    pub struct NativeHintBitsSubEx;
    pub struct NativeHintLoadSubEx;

    impl<F: Field> PhantomSubExecutor<F> for NativeHintInputSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            assert!(streams.hint_stream.is_empty());
            streams
                .hint_stream
                .push_back(F::from_canonical_usize(hint.len()));
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: Field, const N: usize> PhantomSubExecutor<F> for NativeHintSliceSubEx<N> {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            assert!(streams.hint_stream.is_empty());
            assert_eq!(hint.len(), N);
            streams.hint_stream = hint.into();
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativePrintSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            _: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            _: u32,
            c_upper: u16,
        ) -> eyre::Result<()> {
            // TODO: this check should be performed statically on the program before execution
            assert!(
                (c_upper as usize) < memory.memory.config.len(),
                "c_upper out of bounds"
            );
            // SAFETY:
            // - F is stack-allocated repr(C) or repr(transparent)
            // - `c_upper` is a valid address space
            let [value] = unsafe { memory.read::<F, 1>(c_upper as u32, a) };
            println!("{value}");
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativeHintBitsSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            len: u32,
            c_upper: u16,
        ) -> eyre::Result<()> {
            // TODO: this check should be performed statically on the program before execution
            assert!(
                (c_upper as usize) < memory.memory.config.len(),
                "c_upper out of bounds"
            );
            // SAFETY:
            // - F is stack-allocated repr(C) or repr(transparent)
            // - `c_upper` is a valid address space
            let [val] = unsafe { memory.read::<F, 1>(c_upper as u32, a) };
            let mut val = val.as_canonical_u32();

            assert!(streams.hint_stream.is_empty());
            for _ in 0..len {
                streams
                    .hint_stream
                    .push_back(F::from_canonical_u32(val & 1));
                val >>= 1;
            }
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativeHintLoadSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let payload = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            let id = streams.hint_space.len();
            streams.hint_space.push(payload);
            // Hint stream should have already been consumed.
            assert!(streams.hint_stream.is_empty());
            streams.hint_stream.push_back(F::from_canonical_usize(id));
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CastFExtension;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum CastFExtensionExecutor {
    CastF(CastFExecutor),
}

impl<F: PrimeField32> VmExecutionExtension<F> for CastFExtension {
    type Executor = CastFExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, CastFExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let castf = CastFExecutor::new(ConvertAdapterExecutor::new());
        inventory.add_executor(castf, [CastfOpcode::CASTF.global_opcode()])?;
        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for CastFExtension {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;

        let castf = CastFAir::new(
            ConvertAdapterAir::new(exec_bridge, memory_bridge),
            CastFCoreAir::new(range_checker),
        );
        inventory.add_air(castf);
        Ok(())
    }
}

impl<E, SC, RA> VmProverExtension<E, RA, CastFExtension> for NativeCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &CastFExtension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        inventory.next_air::<CastFAir>()?;
        let castf = CastFChip::new(
            CastFCoreFiller::new(ConvertAdapterFiller::new(), range_checker),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(castf);

        Ok(())
    }
}

// Pre-computed maximum trace heights for NativeConfig. Found by doubling
// the actual trace heights of kitchen-sink leaf verification (except for
// VariableRangeChecker, which has a fixed height).
pub const NATIVE_MAX_TRACE_HEIGHTS: &[u32] = &[
    4194304, 4, 128, 2097152, 8388608, 4194304, 262144, 2097152, 16777216, 2097152, 8388608,
    262144, 2097152, 1048576, 4194304, 65536, 262144,
];
