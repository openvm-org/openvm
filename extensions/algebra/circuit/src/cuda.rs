use std::sync::Arc;

use derive_new::new;
use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit::{
    arch::{AdapterCoreLayout, DenseRecordArena, EmptyAdapterCoreLayout, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer, error::CudaError};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::FieldExpressionMetadata;
use openvm_rv32_adapters::{
    Rv32IsEqualModAdapterCols, Rv32IsEqualModAdapterExecutor, Rv32IsEqualModAdapterRecord,
    Rv32VecHeapAdapterCols, Rv32VecHeapAdapterExecutor,
};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};

use crate::{
    fp2_chip::Fp2Chip,
    modular_chip::{ModularChip, ModularIsEqualCoreCols, ModularIsEqualRecord},
    AlgebraRecord,
};

mod cuda_abi {
    use super::*;

    unsafe extern "C" {
        fn launch_modular_addsub_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            record_stride: usize,
            d_prime: *const u8,
            prime_limb_count: u32,
            d_barrett_mu: *const u8,
            q_limbs: u32,
            carry_limbs: u32,
            add_opcode: u32,
            sub_opcode: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;

        fn launch_modular_muldiv_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            record_stride: usize,
            d_prime: *const u8,
            prime_limb_count: u32,
            d_barrett_mu: *const u8,
            q_limbs: u32,
            carry_limbs: u32,
            mul_opcode: u32,
            div_opcode: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;

        fn _modular_is_equal_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            record_len: usize,
            d_modulus: *const u8,
            total_limbs: usize,
            num_lanes: usize,
            lane_size: usize,
            d_range_ctr: *mut u32,
            range_bins: usize,
            d_bitwise_lut: *mut u32,
            bitwise_num_bits: usize,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;

        fn launch_fp2_addsub_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            record_stride: usize,
            d_prime: *const u8,
            prime_limb_count: u32,
            q0_limbs: u32,
            q1_limbs: u32,
            c0_limbs: u32,
            c1_limbs: u32,
            add_opcode: u32,
            sub_opcode: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;

        fn launch_fp2_muldiv_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            record_stride: usize,
            d_prime: *const u8,
            prime_limb_count: u32,
            d_barrett_mu: *const u8,
            q0_limbs: u32,
            q1_limbs: u32,
            c0_limbs: u32,
            c1_limbs: u32,
            mul_opcode: u32,
            div_opcode: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn addsub_tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        d_prime: &DeviceBuffer<u8>,
        prime_limb_count: u32,
        d_barrett_mu: &DeviceBuffer<u8>,
        q_limbs: u32,
        carry_limbs: u32,
        add_opcode: u32,
        sub_opcode: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_modular_addsub_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            d_records.as_ptr(),
            num_records,
            record_stride,
            d_prime.as_ptr(),
            prime_limb_count,
            d_barrett_mu.as_ptr(),
            q_limbs,
            carry_limbs,
            add_opcode,
            sub_opcode,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn muldiv_tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        d_prime: &DeviceBuffer<u8>,
        prime_limb_count: u32,
        d_barrett_mu: &DeviceBuffer<u8>,
        q_limbs: u32,
        carry_limbs: u32,
        mul_opcode: u32,
        div_opcode: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_modular_muldiv_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            d_records.as_ptr(),
            num_records,
            record_stride,
            d_prime.as_ptr(),
            prime_limb_count,
            d_barrett_mu.as_ptr(),
            q_limbs,
            carry_limbs,
            mul_opcode,
            div_opcode,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn is_eq_tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        trace_width: usize,
        d_records: &DeviceBuffer<u8>,
        record_len: usize,
        d_modulus: &DeviceBuffer<u8>,
        total_limbs: usize,
        num_lanes: usize,
        lane_size: usize,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = _modular_is_equal_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            trace_width,
            d_records.as_ptr(),
            record_len,
            d_modulus.as_ptr(),
            total_limbs,
            num_lanes,
            lane_size,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len(),
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp2_addsub_tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        d_prime: &DeviceBuffer<u8>,
        prime_limb_count: u32,
        q_limbs: [u32; 2],
        carry_limbs: [u32; 2],
        add_opcode: u32,
        sub_opcode: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_fp2_addsub_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            d_records.as_ptr(),
            num_records,
            record_stride,
            d_prime.as_ptr(),
            prime_limb_count,
            q_limbs[0],
            q_limbs[1],
            carry_limbs[0],
            carry_limbs[1],
            add_opcode,
            sub_opcode,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp2_muldiv_tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        d_prime: &DeviceBuffer<u8>,
        prime_limb_count: u32,
        d_barrett_mu: &DeviceBuffer<u8>,
        q_limbs: [u32; 2],
        carry_limbs: [u32; 2],
        mul_opcode: u32,
        div_opcode: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_fp2_muldiv_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            d_records.as_ptr(),
            num_records,
            record_stride,
            d_prime.as_ptr(),
            prime_limb_count,
            d_barrett_mu.as_ptr(),
            q_limbs[0],
            q_limbs[1],
            carry_limbs[0],
            carry_limbs[1],
            mul_opcode,
            div_opcode,
            d_range_checker.as_mut_ptr() as *mut u32,
            d_range_checker.len() as u32,
            d_bitwise_lookup.as_mut_ptr() as *mut u32,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits,
        );
        CudaError::from_result(result)
    }
}

fn bigint_to_padded_le_bytes(value: &BigUint, count: usize) -> Vec<u8> {
    let mut bytes = value.to_bytes_le();
    bytes.resize(count, 0);
    bytes
}

#[derive(new)]
pub struct ModularAddSubChipGpu<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: ModularChip<F, BLOCKS, BLOCK_SIZE>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for ModularAddSubChipGpu<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, _>::get_aligned_record_size(
                &layout,
            );
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let trace_height = next_power_of_two_or_zero(num_records);
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let core_width = BaseAir::<F>::width(&self.cpu.inner.expr);
        let trace_width = adapter_width + core_width;

        let builder = &self.cpu.inner.expr.builder;
        assert_eq!(builder.num_variables, 1);
        assert_eq!(builder.q_limbs.len(), 1);
        assert_eq!(builder.carry_limbs.len(), 1);

        let prime_limb_count = builder.prime_limbs.len();
        let prime_bytes = bigint_to_padded_le_bytes(&builder.prime, prime_limb_count);
        let mu_big = (BigUint::one() << (16 * prime_limb_count)) / &builder.prime;
        let mu_bytes = bigint_to_padded_le_bytes(&mu_big, 2 * prime_limb_count);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_prime = prime_bytes.to_device().unwrap();
        let d_barrett_mu = mu_bytes.to_device().unwrap();

        let add_opcode = self.cpu.inner.local_opcode_idx[0] as u32;
        let sub_opcode = self.cpu.inner.local_opcode_idx[1] as u32;

        unsafe {
            cuda_abi::addsub_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
                record_size,
                &d_prime,
                prime_limb_count as u32,
                &d_barrett_mu,
                builder.q_limbs[0] as u32,
                builder.carry_limbs[0] as u32,
                add_opcode,
                sub_opcode,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[derive(new)]
pub struct ModularMulDivChipGpu<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: ModularChip<F, BLOCKS, BLOCK_SIZE>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for ModularMulDivChipGpu<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, _>::get_aligned_record_size(
                &layout,
            );
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let trace_height = next_power_of_two_or_zero(num_records);
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let core_width = BaseAir::<F>::width(&self.cpu.inner.expr);
        let trace_width = adapter_width + core_width;

        let builder = &self.cpu.inner.expr.builder;
        assert_eq!(builder.num_variables, 1);
        assert_eq!(builder.q_limbs.len(), 1);
        assert_eq!(builder.carry_limbs.len(), 1);

        let prime_limb_count = builder.prime_limbs.len();
        let prime_bytes = bigint_to_padded_le_bytes(&builder.prime, prime_limb_count);
        let mu_big = (BigUint::one() << (16 * prime_limb_count)) / &builder.prime;
        let mu_bytes = bigint_to_padded_le_bytes(&mu_big, 2 * prime_limb_count);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_prime = prime_bytes.to_device().unwrap();
        let d_barrett_mu = mu_bytes.to_device().unwrap();

        let mul_opcode = self.cpu.inner.local_opcode_idx[0] as u32;
        let div_opcode = self.cpu.inner.local_opcode_idx[1] as u32;

        unsafe {
            cuda_abi::muldiv_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
                record_size,
                &d_prime,
                prime_limb_count as u32,
                &d_barrett_mu,
                builder.q_limbs[0] as u32,
                builder.carry_limbs[0] as u32,
                mul_opcode,
                div_opcode,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

pub struct ModularIsEqualChipGpu<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> {
    _marker: std::marker::PhantomData<F>,
    modulus_limbs: [u8; TOTAL_LIMBS],
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<F, const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    ModularIsEqualChipGpu<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    pub fn new(
        modulus_limbs: [u8; TOTAL_LIMBS],
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
    ) -> Self {
        Self {
            _marker: std::marker::PhantomData,
            modulus_limbs,
            range_checker,
            bitwise_lookup,
            pointer_max_bits,
            timestamp_max_bits,
        }
    }
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    Chip<DenseRecordArena, GpuBackend>
    for ModularIsEqualChipGpu<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let layout = EmptyAdapterCoreLayout::<
            F,
            Rv32IsEqualModAdapterExecutor<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        >::new();

        let record_size = RecordSeeker::<
            DenseRecordArena,
            (
                &mut Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
                &mut ModularIsEqualRecord<TOTAL_LIMBS>,
            ),
            _,
        >::get_aligned_record_size(&layout);
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace_width = Rv32IsEqualModAdapterCols::<F, 2, NUM_LANES, LANE_SIZE>::width()
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_modulus = self.modulus_limbs.to_vec().to_device().unwrap();

        unsafe {
            cuda_abi::is_eq_tracegen(
                d_trace.buffer(),
                trace_height,
                trace_width,
                &d_records,
                records.len(),
                &d_modulus,
                TOTAL_LIMBS,
                NUM_LANES,
                LANE_SIZE,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[derive(new)]
pub struct Fp2AddSubChipGpu<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for Fp2AddSubChipGpu<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, _>::get_aligned_record_size(
                &layout,
            );
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let trace_height = next_power_of_two_or_zero(num_records);
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let core_width = BaseAir::<F>::width(&self.cpu.inner.expr);
        let trace_width = adapter_width + core_width;

        let builder = &self.cpu.inner.expr.builder;
        assert_eq!(builder.num_variables, 2);
        assert_eq!(builder.q_limbs.len(), 2);
        assert_eq!(builder.carry_limbs.len(), 2);

        let prime_limb_count = builder.prime_limbs.len();
        let prime_bytes = bigint_to_padded_le_bytes(&builder.prime, prime_limb_count);
        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_prime = prime_bytes.to_device().unwrap();

        unsafe {
            cuda_abi::fp2_addsub_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
                record_size,
                &d_prime,
                prime_limb_count as u32,
                [builder.q_limbs[0] as u32, builder.q_limbs[1] as u32],
                [builder.carry_limbs[0] as u32, builder.carry_limbs[1] as u32],
                self.cpu.inner.local_opcode_idx[0] as u32,
                self.cpu.inner.local_opcode_idx[1] as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[derive(new)]
pub struct Fp2MulDivChipGpu<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for Fp2MulDivChipGpu<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, _>::get_aligned_record_size(
                &layout,
            );
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let trace_height = next_power_of_two_or_zero(num_records);
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let core_width = BaseAir::<F>::width(&self.cpu.inner.expr);
        let trace_width = adapter_width + core_width;

        let builder = &self.cpu.inner.expr.builder;
        assert_eq!(builder.num_variables, 2);
        assert_eq!(builder.q_limbs.len(), 2);
        assert_eq!(builder.carry_limbs.len(), 2);

        let prime_limb_count = builder.prime_limbs.len();
        let prime_bytes = bigint_to_padded_le_bytes(&builder.prime, prime_limb_count);
        let mu_big = (BigUint::one() << (16 * prime_limb_count)) / &builder.prime;
        let mu_bytes = bigint_to_padded_le_bytes(&mu_big, 2 * prime_limb_count);
        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_prime = prime_bytes.to_device().unwrap();
        let d_barrett_mu = mu_bytes.to_device().unwrap();

        unsafe {
            cuda_abi::fp2_muldiv_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
                record_size,
                &d_prime,
                prime_limb_count as u32,
                &d_barrett_mu,
                [builder.q_limbs[0] as u32, builder.q_limbs[1] as u32],
                [builder.carry_limbs[0] as u32, builder.carry_limbs[1] as u32],
                self.cpu.inner.local_opcode_idx[0] as u32,
                self.cpu.inner.local_opcode_idx[1] as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
