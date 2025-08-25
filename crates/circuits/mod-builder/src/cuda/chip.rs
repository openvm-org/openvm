#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::{collections::HashMap, sync::Arc};

use cuda_runtime_sys::{cudaDeviceSetLimit, cudaLimit};
use num_bigint::BigUint;
use num_traits::{FromBytes, One};
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, types::F};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::p3_air::BaseAir;

use crate::{
    cuda::{
        constants::{ExprType, LIMB_BITS, MAX_LIMBS},
        expr_op::ExprOp,
    },
    cuda_abi::field_expression::tracegen,
    utils::biguint_to_limbs_vec,
    ExprMeta, ExprNode, FieldExprMeta, FieldExpressionChipGPU, FieldExpressionCoreAir,
    SymbolicExpr,
};

impl FieldExpressionChipGPU {
    pub fn new(
        air: FieldExpressionCoreAir,
        records: DeviceBuffer<u8>,
        num_records: usize,
        record_stride: usize,
        adapter_width: usize,
        adapter_blocks: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<LIMB_BITS>>,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
    ) -> Self {
        let num_inputs = air.num_inputs() as u32;
        let num_vars = air.num_vars() as u32;
        let num_u32_flags = air.num_flags() as u32;
        let core_width = BaseAir::<F>::width(&air) as u32;
        let trace_width = adapter_width as u32 + core_width;

        let num_limbs = air.expr.canonical_num_limbs() as u32;
        let limb_bits = air.expr.canonical_limb_bits() as u32;

        let prime_limbs = air
            .expr
            .builder
            .prime_limbs
            .iter()
            .map(|&x| x as u8)
            .collect::<Vec<_>>();

        // Pad prime_limbs to next valid size (32 or 48)
        let padded_limbs_len = if air.expr.builder.prime_limbs.len() <= 32 {
            32
        } else {
            48
        };
        let mut padded_prime_limbs = air
            .expr
            .builder
            .prime_limbs
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<_>>();
        padded_prime_limbs.resize(padded_limbs_len, 0u32);

        let prime_limbs_buf = padded_prime_limbs.to_device().unwrap();

        // Compute Barrett mu constant
        let p_big: BigUint = BigUint::from_le_bytes(&prime_limbs);
        let actual_limbs = air.expr.builder.prime_limbs.len();
        let two_n_bits = 2 * actual_limbs * limb_bits as usize;
        let b2n = BigUint::one() << two_n_bits;
        let mu_big = &b2n / &p_big;
        let mu_limbs = biguint_to_limbs_vec(&mu_big, 2 * MAX_LIMBS);
        let barrett_mu_buf = mu_limbs.to_device().unwrap();

        let (
            expr_meta,
            compute_expr_ops_buf,
            compute_roots_buf,
            constraint_expr_ops_buf,
            constraint_roots_buf,
            constants_buf,
            const_limb_counts_buf,
            q_limb_counts_buf,
            carry_limb_counts_buf,
            ast_depth,
            max_q_count,
        ) = Self::build_expr_meta(
            &air,
            num_vars,
            num_limbs,
            limb_bits,
            &prime_limbs_buf,
            &barrett_mu_buf,
        );

        let local_opcode_idx_buf = air
            .local_opcode_idx
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();
        let opcode_flag_idx_buf = if air.opcode_flag_idx.is_empty() {
            DeviceBuffer::new()
        } else {
            air.opcode_flag_idx
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<_>>()
                .to_device()
                .unwrap()
        };
        let output_indices_buf = if air.output_indices().is_empty() {
            DeviceBuffer::new()
        } else {
            air.output_indices()
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<_>>()
                .to_device()
                .unwrap()
        };

        let input_limbs_offset = std::mem::size_of::<u8>();

        let meta_host = FieldExprMeta {
            num_inputs,
            num_u32_flags,
            num_limbs,
            limb_bits,
            adapter_blocks: adapter_blocks as u32,
            adapter_width: adapter_width as u32,
            core_width,
            trace_width,
            local_opcode_idx: local_opcode_idx_buf.as_ptr(),
            opcode_flag_idx: opcode_flag_idx_buf.as_ptr(),
            output_indices: output_indices_buf.as_ptr(),
            num_local_opcodes: air.local_opcode_idx.len() as u32,
            num_output_indices: air.output_indices().len() as u32,
            record_stride: record_stride as u32,
            input_limbs_offset: input_limbs_offset as u32,
            q_limb_counts: q_limb_counts_buf.as_ptr(),
            carry_limb_counts: carry_limb_counts_buf.as_ptr(),
            compute_expr_ops: compute_expr_ops_buf.as_ptr() as *const ExprOp,
            compute_root_indices: compute_roots_buf.as_ptr(),
            constraint_expr_ops: constraint_expr_ops_buf.as_ptr() as *const ExprOp,
            constraint_root_indices: constraint_roots_buf.as_ptr(),
            max_q_count,
            expr_meta,
            max_ast_depth: ast_depth,
        };

        let meta = vec![meta_host].to_device().unwrap();

        Self {
            air,
            records: Arc::new(records),
            num_records,
            record_stride,
            total_trace_width: trace_width as usize,
            meta,
            local_opcode_idx_buf,
            opcode_flag_idx_buf,
            output_indices_buf,
            prime_limbs_buf,
            compute_expr_ops_buf,
            compute_roots_buf,
            constraint_expr_ops_buf,
            constraint_roots_buf,
            constants_buf,
            const_limb_counts_buf,
            q_limb_counts_buf,
            carry_limb_counts_buf,
            barrett_mu_buf,
            range_checker,
            bitwise_lookup,
            pointer_max_bits,
            timestamp_max_bits,
        }
    }

    fn build_expr_meta(
        air: &FieldExpressionCoreAir,
        num_vars: u32,
        num_limbs: u32,
        limb_bits: u32,
        prime_limbs_buf: &DeviceBuffer<u32>,
        barrett_mu_buf: &DeviceBuffer<u8>,
    ) -> (
        ExprMeta,
        DeviceBuffer<u128>,
        DeviceBuffer<u32>,
        DeviceBuffer<u128>,
        DeviceBuffer<u32>,
        DeviceBuffer<u32>,
        DeviceBuffer<u32>,
        DeviceBuffer<u32>,
        DeviceBuffer<u32>,
        u32,
        u32,
    ) {
        // Build compute expressions AST
        let mut compute_expr_pool = Vec::new();
        let mut compute_node_map = HashMap::new();
        let mut compute_root_indices = Vec::with_capacity(air.expr.builder.computes.len());
        let mut max_compute_depth = 0;
        for compute_expr in &air.expr.builder.computes {
            let depth = Self::calculate_ast_depth(compute_expr);
            max_compute_depth = max_compute_depth.max(depth);
            let root =
                Self::add_expr_to_pool(compute_expr, &mut compute_expr_pool, &mut compute_node_map);
            compute_root_indices.push(root);
        }

        // Build constraint expressions AST
        let mut constraint_expr_pool = Vec::new();
        let mut constraint_node_map = HashMap::new();
        let mut constraint_root_indices = Vec::with_capacity(air.expr.builder.constraints.len());
        let mut max_constraint_depth = 0;
        for constraint_expr in &air.expr.builder.constraints {
            let depth = Self::calculate_ast_depth(constraint_expr);
            max_constraint_depth = max_constraint_depth.max(depth);
            let root = Self::add_expr_to_pool(
                constraint_expr,
                &mut constraint_expr_pool,
                &mut constraint_node_map,
            );
            constraint_root_indices.push(root);
        }

        let ast_depth = max_compute_depth.max(max_constraint_depth);

        // Extract constants
        let mut constants = Vec::new();
        let mut const_limb_counts = Vec::new();
        for (_const_val, const_limbs) in &air.expr.builder.constants {
            const_limb_counts.push(const_limbs.len() as u32);
            for &limb in const_limbs {
                constants.push(limb as u32);
            }
        }
        let q_counts: Vec<u32> = air.expr.builder.q_limbs.iter().map(|&x| x as u32).collect();
        let carry_counts: Vec<u32> = air
            .expr
            .builder
            .carry_limbs
            .iter()
            .map(|&x| x as u32)
            .collect();

        let max_q_count = if q_counts.is_empty() {
            num_limbs + 1 // Default fallback
        } else {
            *q_counts.iter().max().unwrap()
        };

        let compute_expr_ops_u128: Vec<u128> = compute_expr_pool
            .iter()
            .map(|n| ExprOp::from_node(n).0)
            .collect();
        let constraint_expr_ops_u128: Vec<u128> = constraint_expr_pool
            .iter()
            .map(|n| ExprOp::from_node(n).0)
            .collect();

        let compute_expr_ops_buf = compute_expr_ops_u128.to_device().unwrap();
        let constraint_expr_ops_buf = constraint_expr_ops_u128.to_device().unwrap();

        let compute_roots_buf = compute_root_indices.to_device().unwrap();
        let constraint_roots_buf = constraint_root_indices.to_device().unwrap();

        let constants_buf = if constants.is_empty() {
            DeviceBuffer::new()
        } else {
            constants.to_device().unwrap()
        };
        let const_limb_counts_buf = if const_limb_counts.is_empty() {
            DeviceBuffer::new()
        } else {
            const_limb_counts.to_device().unwrap()
        };
        let q_limb_counts_buf = if q_counts.is_empty() {
            DeviceBuffer::new()
        } else {
            q_counts.to_device().unwrap()
        };
        let carry_limb_counts_buf = if carry_counts.is_empty() {
            DeviceBuffer::new()
        } else {
            carry_counts.to_device().unwrap()
        };

        let expr_meta = ExprMeta {
            constants: constants_buf.as_ptr(),
            const_limb_counts: const_limb_counts_buf.as_ptr(),
            q_limb_counts: q_limb_counts_buf.as_ptr(),
            carry_limb_counts: carry_limb_counts_buf.as_ptr(),
            num_vars,
            num_constants: air.expr.builder.constants.len() as u32,
            expr_pool_size: compute_expr_pool.len() as u32 + constraint_expr_pool.len() as u32,
            prime_limbs: prime_limbs_buf.as_ptr(),
            prime_limb_count: num_limbs,
            limb_bits,
            barrett_mu: barrett_mu_buf.as_ptr(),
        };

        (
            expr_meta,
            compute_expr_ops_buf,
            compute_roots_buf,
            constraint_expr_ops_buf,
            constraint_roots_buf,
            constants_buf,
            const_limb_counts_buf,
            q_limb_counts_buf,
            carry_limb_counts_buf,
            ast_depth,
            max_q_count,
        )
    }

    fn calculate_ast_depth(expr: &SymbolicExpr) -> u32 {
        match expr {
            SymbolicExpr::Input(_) | SymbolicExpr::Var(_) | SymbolicExpr::Const(_, _, _) => 1,
            SymbolicExpr::Add(left, right)
            | SymbolicExpr::Sub(left, right)
            | SymbolicExpr::Mul(left, right)
            | SymbolicExpr::Div(left, right) => {
                1 + Self::calculate_ast_depth(left).max(Self::calculate_ast_depth(right))
            }
            SymbolicExpr::IntAdd(child, _) | SymbolicExpr::IntMul(child, _) => {
                1 + Self::calculate_ast_depth(child)
            }
            SymbolicExpr::Select(_, if_true, if_false) => {
                1 + Self::calculate_ast_depth(if_true).max(Self::calculate_ast_depth(if_false))
            }
        }
    }

    fn convert_to_expr_node(
        expr: &SymbolicExpr,
        expr_pool: &mut Vec<ExprNode>,
        node_map: &mut HashMap<String, u32>,
    ) -> ExprNode {
        match expr {
            SymbolicExpr::Input(idx) => ExprNode {
                r#type: ExprType::Input as u32,
                data: [*idx as u32, 0, 0],
            },
            SymbolicExpr::Var(idx) => ExprNode {
                r#type: ExprType::Var as u32,
                data: [*idx as u32, 0, 0],
            },
            SymbolicExpr::Const(idx, _val, _limbs) => ExprNode {
                r#type: ExprType::Const as u32,
                data: [*idx as u32, 0, 0],
            },
            SymbolicExpr::Add(left, right) => {
                let left_idx = Self::add_expr_to_pool(left, expr_pool, node_map);
                let right_idx = Self::add_expr_to_pool(right, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::Add as u32,
                    data: [left_idx, right_idx, 0],
                }
            }
            SymbolicExpr::Sub(left, right) => {
                let left_idx = Self::add_expr_to_pool(left, expr_pool, node_map);
                let right_idx = Self::add_expr_to_pool(right, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::Sub as u32,
                    data: [left_idx, right_idx, 0],
                }
            }
            SymbolicExpr::Mul(left, right) => {
                let left_idx = Self::add_expr_to_pool(left, expr_pool, node_map);
                let right_idx = Self::add_expr_to_pool(right, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::Mul as u32,
                    data: [left_idx, right_idx, 0],
                }
            }
            SymbolicExpr::Div(left, right) => {
                let left_idx = Self::add_expr_to_pool(left, expr_pool, node_map);
                let right_idx = Self::add_expr_to_pool(right, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::Div as u32,
                    data: [left_idx, right_idx, 0],
                }
            }
            SymbolicExpr::IntAdd(child, scalar) => {
                let child_idx = Self::add_expr_to_pool(child, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::IntAdd as u32,
                    data: [child_idx, *scalar as u32, 0],
                }
            }
            SymbolicExpr::IntMul(child, scalar) => {
                let child_idx = Self::add_expr_to_pool(child, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::IntMul as u32,
                    data: [child_idx, *scalar as u32, 0],
                }
            }
            SymbolicExpr::Select(flag_idx, if_true, if_false) => {
                let true_idx = Self::add_expr_to_pool(if_true, expr_pool, node_map);
                let false_idx = Self::add_expr_to_pool(if_false, expr_pool, node_map);
                ExprNode {
                    r#type: ExprType::Select as u32,
                    data: [*flag_idx as u32, true_idx, false_idx],
                }
            }
        }
    }

    fn add_expr_to_pool(
        expr: &SymbolicExpr,
        expr_pool: &mut Vec<ExprNode>,
        node_map: &mut HashMap<String, u32>,
    ) -> u32 {
        let expr_str = format!("{:?}", expr); // Simple deduplication key

        if let Some(&existing_idx) = node_map.get(&expr_str) {
            return existing_idx;
        }

        // Create the node based on expression type
        let node = Self::convert_to_expr_node(expr, expr_pool, node_map);

        let idx = expr_pool.len() as u32;
        node_map.insert(expr_str, idx);

        expr_pool.push(node);
        idx
    }

    pub fn generate_field_trace(&self) -> DeviceMatrix<F> {
        let padded_height = next_power_of_two_or_zero(self.num_records);
        let mat = DeviceMatrix::with_capacity(padded_height, self.total_trace_width);

        let meta_host = self.meta.to_host().unwrap()[0].clone();
        let input_size = meta_host.num_inputs * meta_host.num_limbs;
        let var_size = meta_host.expr_meta.num_vars * meta_host.num_limbs;
        let carry_counts = self.carry_limb_counts_buf.to_host().unwrap();
        let total_carry_count: u32 = carry_counts.iter().sum::<u32>();

        // size in bytes
        let workspace_per_thread = (input_size + var_size + total_carry_count)
            * (size_of::<u32>() as u32)
            + meta_host.num_u32_flags;

        // Align workspace size to 16 bytes for CUDA alignment requirements
        let workspace_per_thread = workspace_per_thread.next_multiple_of(16);

        // Allocate workspace for all threads
        let total_workspace_size = (workspace_per_thread as usize)
            .checked_mul(padded_height)
            .unwrap();
        let workspace = DeviceBuffer::<u8>::with_capacity(total_workspace_size);

        unsafe {
            cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 48 * 1024);
            tracegen(
                &self.records,
                mat.buffer(),
                &self.meta,
                self.num_records,
                self.record_stride,
                self.total_trace_width,
                padded_height,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                LIMB_BITS as u32,
                self.pointer_max_bits,
                self.timestamp_max_bits,
                workspace.as_ptr(),
                workspace_per_thread,
            )
            .unwrap();
        }
        mat
    }
}
