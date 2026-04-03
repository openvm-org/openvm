use std::sync::Arc;

use derive_new::new;
use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit::{
    arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer, error::CudaError};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::FieldExpressionMetadata;
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterExecutor};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};

use crate::{EccRecord, WeierstrassChip};

const CUDA_PREPARED_PRIME_MAX_LIMBS: usize = 49;

#[repr(C)]
#[derive(Clone, Copy)]
struct BigUintGpuLayout {
    limbs: [u8; CUDA_PREPARED_PRIME_MAX_LIMBS],
    padding: [u8; 3],
    num_limbs: u32,
    limb_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OverflowIntLayout {
    limbs: [i32; CUDA_PREPARED_PRIME_MAX_LIMBS],
    num_limbs: u32,
    limb_bits: u32,
    limb_max_abs: u32,
    max_overflow_bits: u32,
}

mod cuda_abi {
    use super::*;

    unsafe extern "C" {
        fn launch_ec_add_ne_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            record_stride: usize,
            d_prime: *const u8,
            d_prepared_prime: *mut BigUintGpuLayout,
            d_prepared_prime_overflow: *mut OverflowIntLayout,
            prime_limb_count: u32,
            d_barrett_mu: *const u8,
            q0_limbs: u32,
            q1_limbs: u32,
            q2_limbs: u32,
            c0_limbs: u32,
            c1_limbs: u32,
            c2_limbs: u32,
            setup_opcode: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;

        fn launch_ec_double_tracegen(
            d_trace: *mut F,
            trace_height: usize,
            d_records: *const u8,
            num_records: usize,
            record_stride: usize,
            d_prime: *const u8,
            d_prepared_prime: *mut BigUintGpuLayout,
            d_prepared_prime_overflow: *mut OverflowIntLayout,
            prime_limb_count: u32,
            d_a: *const u8,
            a_limb_count: u32,
            d_barrett_mu: *const u8,
            q0_limbs: u32,
            q1_limbs: u32,
            q2_limbs: u32,
            q3_limbs: u32,
            c0_limbs: u32,
            c1_limbs: u32,
            c2_limbs: u32,
            c3_limbs: u32,
            num_variables: u32,
            setup_opcode: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            pointer_max_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        d_prime: &DeviceBuffer<u8>,
        d_prepared_prime: &DeviceBuffer<BigUintGpuLayout>,
        d_prepared_prime_overflow: &DeviceBuffer<OverflowIntLayout>,
        prime_limb_count: u32,
        d_barrett_mu: &DeviceBuffer<u8>,
        q_limbs: [u32; 3],
        carry_limbs: [u32; 3],
        setup_opcode: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_ec_add_ne_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            d_records.as_ptr(),
            num_records,
            record_stride,
            d_prime.as_ptr(),
            d_prepared_prime.as_mut_ptr(),
            d_prepared_prime_overflow.as_mut_ptr(),
            prime_limb_count,
            d_barrett_mu.as_ptr(),
            q_limbs[0],
            q_limbs[1],
            q_limbs[2],
            carry_limbs[0],
            carry_limbs[1],
            carry_limbs[2],
            setup_opcode,
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
    pub unsafe fn double_tracegen(
        d_trace: &DeviceBuffer<F>,
        trace_height: usize,
        d_records: &DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        d_prime: &DeviceBuffer<u8>,
        d_prepared_prime: &DeviceBuffer<BigUintGpuLayout>,
        d_prepared_prime_overflow: &DeviceBuffer<OverflowIntLayout>,
        prime_limb_count: u32,
        d_a: &DeviceBuffer<u8>,
        a_limb_count: u32,
        d_barrett_mu: &DeviceBuffer<u8>,
        q_limbs: [u32; 4],
        carry_limbs: [u32; 4],
        num_variables: u32,
        setup_opcode: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let result = launch_ec_double_tracegen(
            d_trace.as_mut_ptr(),
            trace_height,
            d_records.as_ptr(),
            num_records,
            record_stride,
            d_prime.as_ptr(),
            d_prepared_prime.as_mut_ptr(),
            d_prepared_prime_overflow.as_mut_ptr(),
            prime_limb_count,
            d_a.as_ptr(),
            a_limb_count,
            d_barrett_mu.as_ptr(),
            q_limbs[0],
            q_limbs[1],
            q_limbs[2],
            q_limbs[3],
            carry_limbs[0],
            carry_limbs[1],
            carry_limbs[2],
            carry_limbs[3],
            num_variables,
            setup_opcode,
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
pub struct EcAddNeChipGpu<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for EcAddNeChipGpu<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, EccRecord<2, BLOCKS, BLOCK_SIZE>, _>::get_aligned_record_size(
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
        assert_eq!(builder.num_variables, 3);
        assert_eq!(builder.q_limbs.len(), 3);
        assert_eq!(builder.carry_limbs.len(), 3);

        let prime_limb_count = builder.prime_limbs.len();
        let prime_bytes = bigint_to_padded_le_bytes(&builder.prime, prime_limb_count);
        let mu_big = (BigUint::one() << (16 * prime_limb_count)) / &builder.prime;
        let mu_bytes = bigint_to_padded_le_bytes(&mu_big, 2 * prime_limb_count);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_prime = prime_bytes.to_device().unwrap();
        let d_prepared_prime = DeviceBuffer::<BigUintGpuLayout>::with_capacity(1);
        let d_prepared_prime_overflow = DeviceBuffer::<OverflowIntLayout>::with_capacity(1);
        let d_barrett_mu = mu_bytes.to_device().unwrap();

        let q_limbs = [
            builder.q_limbs[0] as u32,
            builder.q_limbs[1] as u32,
            builder.q_limbs[2] as u32,
        ];
        let carry_limbs = [
            builder.carry_limbs[0] as u32,
            builder.carry_limbs[1] as u32,
            builder.carry_limbs[2] as u32,
        ];
        let setup_opcode = self.cpu.inner.local_opcode_idx.last().copied().unwrap_or_default() as u32;

        unsafe {
            cuda_abi::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
                record_size,
                &d_prime,
                &d_prepared_prime,
                &d_prepared_prime_overflow,
                prime_limb_count as u32,
                &d_barrett_mu,
                q_limbs,
                carry_limbs,
                setup_opcode,
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
pub struct EcDoubleChipGpu<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE>,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for EcDoubleChipGpu<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, EccRecord<1, BLOCKS, BLOCK_SIZE>, _>::get_aligned_record_size(
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
            Rv32VecHeapAdapterCols::<F, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let core_width = BaseAir::<F>::width(&self.cpu.inner.expr);
        let trace_width = adapter_width + core_width;

        let builder = &self.cpu.inner.expr.builder;
        assert!((3..=4).contains(&builder.num_variables));
        assert_eq!(builder.q_limbs.len(), builder.num_variables);
        assert_eq!(builder.carry_limbs.len(), builder.num_variables);

        let prime_limb_count = builder.prime_limbs.len();
        let prime_bytes = bigint_to_padded_le_bytes(&builder.prime, prime_limb_count);
        let a_value = self
            .cpu
            .inner
            .expr
            .setup_values
            .first()
            .expect("EcDouble setup value a must exist");
        let a_bytes = bigint_to_padded_le_bytes(a_value, builder.num_limbs);
        let mu_big = (BigUint::one() << (16 * prime_limb_count)) / &builder.prime;
        let mu_bytes = bigint_to_padded_le_bytes(&mu_big, 2 * prime_limb_count);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);
        let d_prime = prime_bytes.to_device().unwrap();
        let d_prepared_prime = DeviceBuffer::<BigUintGpuLayout>::with_capacity(1);
        let d_prepared_prime_overflow = DeviceBuffer::<OverflowIntLayout>::with_capacity(1);
        let d_a = a_bytes.to_device().unwrap();
        let d_barrett_mu = mu_bytes.to_device().unwrap();

        let mut q_limbs = [0u32; 4];
        for (dst, src) in q_limbs.iter_mut().zip(builder.q_limbs.iter()) {
            *dst = *src as u32;
        }
        let mut carry_limbs = [0u32; 4];
        for (dst, src) in carry_limbs.iter_mut().zip(builder.carry_limbs.iter()) {
            *dst = *src as u32;
        }
        let setup_opcode = self.cpu.inner.local_opcode_idx.last().copied().unwrap_or_default() as u32;

        unsafe {
            cuda_abi::double_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                num_records,
                record_size,
                &d_prime,
                &d_prepared_prime,
                &d_prepared_prime_overflow,
                prime_limb_count as u32,
                &d_a,
                builder.num_limbs as u32,
                &d_barrett_mu,
                q_limbs,
                carry_limbs,
                builder.num_variables as u32,
                setup_opcode,
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
