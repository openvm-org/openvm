use std::sync::Arc;

use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_common::d_buffer::DeviceBuffer;

use crate::{
    cuda::{constants::LIMB_BITS, expr_op::ExprOp},
    FieldExpressionCoreAir,
};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ExprMeta {
    pub constants: *const u32,
    pub const_limb_counts: *const u32,
    pub q_limb_counts: *const u32,
    pub carry_limb_counts: *const u32,

    pub num_vars: u32,
    pub num_constants: u32,
    pub expr_pool_size: u32,

    pub prime_limbs: *const u32,
    pub prime_limb_count: u32,
    pub limb_bits: u32,

    pub barrett_mu: *const u8,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ExprNode {
    pub r#type: u32,
    pub data: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct FieldExprMeta {
    pub num_inputs: u32,
    pub num_u32_flags: u32,
    pub num_limbs: u32,
    pub limb_bits: u32,
    pub adapter_blocks: u32,
    pub adapter_width: u32,
    pub core_width: u32,
    pub trace_width: u32,

    pub local_opcode_idx: *const u32,
    pub opcode_flag_idx: *const u32,
    pub output_indices: *const u32,

    pub num_local_opcodes: u32,
    pub num_output_indices: u32,

    pub record_stride: u32,
    pub input_limbs_offset: u32,

    pub q_limb_counts: *const u32,
    pub carry_limb_counts: *const u32,
    pub compute_expr_ops: *const ExprOp,
    pub compute_root_indices: *const u32,
    pub constraint_expr_ops: *const ExprOp,
    pub constraint_root_indices: *const u32,

    pub max_q_count: u32,

    pub expr_meta: ExprMeta,

    pub max_ast_depth: u32,
}

unsafe impl Send for FieldExprMeta {}
unsafe impl Sync for FieldExprMeta {}

pub struct FieldExpressionChipGPU {
    pub air: FieldExpressionCoreAir,
    pub records: Arc<DeviceBuffer<u8>>,
    pub num_records: usize,
    pub record_stride: usize,
    pub total_trace_width: usize,

    // Metadata and device arrays
    pub meta: DeviceBuffer<FieldExprMeta>,
    pub local_opcode_idx_buf: DeviceBuffer<u32>,
    pub opcode_flag_idx_buf: DeviceBuffer<u32>,
    pub output_indices_buf: DeviceBuffer<u32>,

    // Prime modulus for field arithmetic
    pub prime_limbs_buf: DeviceBuffer<u32>,

    // Expression metadata device buffers
    pub compute_expr_ops_buf: DeviceBuffer<u128>,
    pub compute_roots_buf: DeviceBuffer<u32>,
    pub constraint_expr_ops_buf: DeviceBuffer<u128>,
    pub constraint_roots_buf: DeviceBuffer<u32>,
    pub constants_buf: DeviceBuffer<u32>,
    pub const_limb_counts_buf: DeviceBuffer<u32>,
    pub q_limb_counts_buf: DeviceBuffer<u32>,
    pub carry_limb_counts_buf: DeviceBuffer<u32>,
    pub barrett_mu_buf: DeviceBuffer<u8>,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<LIMB_BITS>>,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
}
