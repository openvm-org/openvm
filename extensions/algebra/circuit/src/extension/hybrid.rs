//! Prover extension for the GPU backend; FieldExpr chips do trace generation on GPU.

use std::sync::Arc;

use openvm_circuit::{
    arch::*,
    system::{
        cuda::{
            extensions::{get_inventory_range_checker, SystemGpuBuilder},
            SystemChipInventoryGPU,
        },
        memory::SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{
    base::DeviceMatrix,
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_mod_circuit_builder::{
    cuda::FieldExprChipGpu, ExprBuilderConfig, FieldExpressionMetadata,
};
use openvm_riscv_adapters::{
    Rv64IsEqualModU16AdapterCols, Rv64IsEqualModU16AdapterRecord, Rv64VecHeapAdapterCols,
    Rv64VecHeapAdapterExecutor,
};
use openvm_riscv_circuit::{adapters::U16_BITS, Rv64ImGpuProverExt};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    fp2_chip::{get_fp2_addsub_chip, get_fp2_muldiv_chip, Fp2Air, Fp2Chip},
    modular_chip::*,
    AlgebraRecord, Fp2Extension, ModularExtension, Rv64ModularConfig, Rv64ModularWithFp2Config,
    FP2_BLOCKS_32, FP2_BLOCKS_48, MODULAR_BLOCKS_32, MODULAR_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_32_U16, NUM_LIMBS_48, NUM_LIMBS_48_U16,
};

pub struct HybridModularChip<F, const BLOCKS: usize> {
    gpu: FieldExprChipGpu,
    _phantom: std::marker::PhantomData<ModularChip<F, BLOCKS>>,
}

impl<const BLOCKS: usize> HybridModularChip<F, BLOCKS> {
    pub fn new(
        cpu: ModularChip<F, BLOCKS>,
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker_gpu: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        let total_input_limbs =
            cpu.inner.num_inputs() * cpu.inner.expr.program().canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>,
        >::new(total_input_limbs));
        let (adapter_size, core_size) =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS>, _>::get_aligned_sizes(
                &layout,
            );
        let gpu = FieldExprChipGpu::new(
            &cpu.inner,
            2,
            BLOCKS,
            Rv64VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS>::width(),
            adapter_size + core_size,
            adapter_size,
            byte_ptr_max_bits as u32,
            timestamp_max_bits as u32,
            range_checker_gpu,
        );
        Self {
            gpu,
            _phantom: std::marker::PhantomData,
        }
    }
}

// GPU tracegen: the field_expr kernel fills adapter + core columns directly from
// the dense records (see openvm_mod_circuit_builder::cuda).
impl<const BLOCKS: usize> Chip<DenseRecordArena, GpuBackend> for HybridModularChip<F, BLOCKS> {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        self.gpu.generate_proving_ctx(arena.allocated())
    }
}

#[derive(Clone, Copy, Default)]
pub struct AlgebraHybridProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, ModularExtension>
    for AlgebraHybridProverExt
{
    fn extend_prover(
        &self,
        extension: &ModularExtension,
        inventory: &mut ChipInventory<SC, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker_gpu = get_inventory_range_checker(inventory);
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        for modulus in extension.supported_moduli.iter() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;

            let modulus_limbs = big_uint_to_limbs(modulus, U16_BITS);

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32>>()?;
                let addsub = get_modular_addsub_chip::<F, MODULAR_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    addsub,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    muldiv,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u16
                    } else {
                        0
                    }
                });
                inventory
                    .next_air::<ModularIsEqualU16Air<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>>()?;
                let is_eq = ModularIsEqualChipGpu::<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>::new(
                    modulus_limbs,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                );
                inventory.add_executor_chip(is_eq);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48>>()?;
                let addsub = get_modular_addsub_chip::<F, MODULAR_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    addsub,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    muldiv,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u16
                    } else {
                        0
                    }
                });
                inventory
                    .next_air::<ModularIsEqualU16Air<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>>()?;
                let is_eq = ModularIsEqualChipGpu::<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>::new(
                    modulus_limbs,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                );
                inventory.add_executor_chip(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

pub struct HybridFp2Chip<F, const BLOCKS: usize> {
    gpu: FieldExprChipGpu,
    _phantom: std::marker::PhantomData<Fp2Chip<F, BLOCKS>>,
}

impl<const BLOCKS: usize> HybridFp2Chip<F, BLOCKS> {
    pub fn new(
        cpu: Fp2Chip<F, BLOCKS>,
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker_gpu: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        let total_input_limbs =
            cpu.inner.num_inputs() * cpu.inner.expr.program().canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>,
        >::new(total_input_limbs));
        let (adapter_size, core_size) =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS>, _>::get_aligned_sizes(
                &layout,
            );
        let gpu = FieldExprChipGpu::new(
            &cpu.inner,
            2,
            BLOCKS,
            Rv64VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS>::width(),
            adapter_size + core_size,
            adapter_size,
            byte_ptr_max_bits as u32,
            timestamp_max_bits as u32,
            range_checker_gpu,
        );
        Self {
            gpu,
            _phantom: std::marker::PhantomData,
        }
    }
}

// GPU tracegen: the field_expr kernel fills adapter + core columns directly from
// the dense records (see openvm_mod_circuit_builder::cuda).
impl<const BLOCKS: usize> Chip<DenseRecordArena, GpuBackend> for HybridFp2Chip<F, BLOCKS> {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        self.gpu.generate_proving_ctx(arena.allocated())
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Fp2Extension>
    for AlgebraHybridProverExt
{
    fn extend_prover(
        &self,
        extension: &Fp2Extension,
        inventory: &mut ChipInventory<SC, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker_gpu = get_inventory_range_checker(inventory);
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        for (_, modulus) in extension.supported_moduli.iter() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32>>()?;
                let addsub = get_fp2_addsub_chip::<F, FP2_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    addsub,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    muldiv,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48>>()?;
                let addsub = get_fp2_addsub_chip::<F, FP2_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    addsub,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    muldiv,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                ));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

/// This builder will do tracegen for the RV64IM extensions on GPU but the modular extensions on
/// CPU.
#[derive(Clone)]
pub struct Rv64ModularHybridBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv64ModularHybridBuilder {
    type VmConfig = Rv64ModularConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64ModularConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.mul, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraHybridProverExt,
            &config.modular,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

/// This builder will do tracegen for the RV64IM extensions on GPU but the modular and complex
/// extensions on CPU.
#[derive(Clone)]
pub struct Rv64ModularWithFp2HybridBuilder;

impl VmBuilder<E> for Rv64ModularWithFp2HybridBuilder {
    type VmConfig = Rv64ModularWithFp2Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64ModularWithFp2Config,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ModularHybridBuilder,
            &config.modular,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraHybridProverExt,
            &config.fp2,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

// ---------------------------------------------------------------------------
// GPU tracegen for ModularIsEqual (dedicated kernel; see cuda/src/modular_is_eq.cu)
// ---------------------------------------------------------------------------

mod is_eq_cuda_abi {
    #![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
    use openvm_cuda_backend::prelude::F;
    use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t};

    macro_rules! declare_is_eq_launcher {
        ($name:ident) => {
            extern "C" {
                fn $name(
                    d_trace: *mut F,
                    height: usize,
                    rows_used: usize,
                    d_records: *const u8,
                    rec_stride: usize,
                    rec_core_offset: usize,
                    d_modulus_limbs: *const u16,
                    d_range_checker: *mut u32,
                    rc_bins: usize,
                    pointer_max_bits: u32,
                    timestamp_max_bits: u32,
                    stream: cudaStream_t,
                ) -> i32;
            }
        };
    }
    declare_is_eq_launcher!(_modular_is_eq_tracegen_l4);
    declare_is_eq_launcher!(_modular_is_eq_tracegen_l6);

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        rows_used: usize,
        d_records: &DeviceBuffer<u8>,
        rec_stride: usize,
        rec_core_offset: usize,
        d_modulus_limbs: &DeviceBuffer<u16>,
        d_range_checker: &DeviceBuffer<F>,
        num_lanes: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        let launcher = match num_lanes {
            4 => _modular_is_eq_tracegen_l4,
            6 => _modular_is_eq_tracegen_l6,
            _ => panic!("unsupported ModularIsEqual num_lanes {num_lanes}"),
        };
        CudaError::from_result(launcher(
            d_trace.as_mut_ptr(),
            height,
            rows_used,
            d_records.as_ptr(),
            rec_stride,
            rec_core_offset,
            d_modulus_limbs.as_ptr(),
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            pointer_max_bits,
            timestamp_max_bits,
            stream,
        ))
    }
}

pub struct ModularIsEqualChipGpu<const NUM_LANES: usize, const TOTAL_LIMBS: usize> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    d_modulus: openvm_cuda_common::d_buffer::DeviceBuffer<u16>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize>
    ModularIsEqualChipGpu<NUM_LANES, TOTAL_LIMBS>
{
    pub fn new(
        modulus_limbs: [u16; TOTAL_LIMBS],
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        use openvm_cuda_common::copy::MemCopyH2D;
        let d_modulus = modulus_limbs
            .as_slice()
            .to_device_on(&range_checker.device_ctx)
            .unwrap();
        Self {
            range_checker,
            d_modulus,
            pointer_max_bits: byte_ptr_max_bits as u32,
            timestamp_max_bits: timestamp_max_bits as u32,
        }
    }
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize> Chip<DenseRecordArena, GpuBackend>
    for ModularIsEqualChipGpu<NUM_LANES, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        use openvm_cuda_common::copy::MemCopyH2D;
        type ARec<const L: usize> = Rv64IsEqualModU16AdapterRecord<2, L>;
        let record_stride = size_of::<(ARec<NUM_LANES>, ModularIsEqualRecord<TOTAL_LIMBS>)>();
        let rec_core_offset = size_of::<ARec<NUM_LANES>>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_stride, 0);
        let rows_used = records.len() / record_stride;
        let height = rows_used.next_power_of_two();
        let width = Rv64IsEqualModU16AdapterCols::<F, 2, NUM_LANES>::width()
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();

        let device_ctx = &self.range_checker.device_ctx;
        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        unsafe {
            is_eq_cuda_abi::tracegen(
                d_trace.buffer(),
                height,
                rows_used,
                &d_records,
                record_stride,
                rec_core_offset,
                &self.d_modulus,
                &self.range_checker.count,
                NUM_LANES,
                self.pointer_max_bits,
                self.timestamp_max_bits,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
