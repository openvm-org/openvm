use openvm_cuda_backend::prelude::{Digest, F};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicExpressionNode,
    },
    keygen::types::StarkVerifyingKey,
};
use p3_field::PrimeCharacteristicRing;

use crate::batch_constraint::expr_eval::{CachedTraceRecord, NodeKind};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlatSymbolicConstraintNode {
    pub kind: u8,
    pub data0: u32,
    pub data1: u32,
    pub data2: u32,
    pub constant: F,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlatInteraction {
    pub count: u32,
    pub message_start: u32,
    pub message_len: u32,
    pub bus_index: u32,
    pub count_weight: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlatSymbolicVariable {
    pub entry_kind: u8,
    pub index: u32,
    pub part_index: u32,
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CachedGpuRecord {
    pub poseidon2_start: [F; POSEIDON2_WIDTH],
    pub is_constraint: bool,
}

pub(crate) fn build_cached_gpu_records(
    cached_trace_record: &CachedTraceRecord,
) -> Option<Vec<CachedGpuRecord>> {
    cached_trace_record
        .dag_commit_info
        .as_ref()
        .map(|dag_commit_info| {
            // We need one Poseidon2 start-state per row of the (power-of-two) trace height, including
            // padding rows. Padding rows have `is_constraint = false`.
            let height = dag_commit_info.poseidon2_inputs.len();
            debug_assert_eq!(
                height,
                cached_trace_record.records.len().next_power_of_two()
            );

            (0..height)
                .map(|row_idx| CachedGpuRecord {
                    poseidon2_start: *dag_commit_info.poseidon2_inputs.get(row_idx).unwrap(),
                    is_constraint: cached_trace_record
                        .records
                        .get(row_idx)
                        .is_some_and(|r| r.is_constraint),
                })
                .collect()
        })
}

pub(super) fn flatten_constraint_node(
    vk: &StarkVerifyingKey<F, Digest>,
    node: &SymbolicExpressionNode<F>,
) -> FlatSymbolicConstraintNode {
    match node {
        SymbolicExpressionNode::Variable(var) => match var.entry {
            Entry::Preprocessed { offset } => FlatSymbolicConstraintNode {
                kind: NodeKind::VarPreprocessed as u8,
                data0: var.index as u32,
                data1: 0,
                data2: offset as u32,
                constant: F::ZERO,
            },
            Entry::Main { part_index, offset } => FlatSymbolicConstraintNode {
                kind: NodeKind::VarMain as u8,
                data0: var.index as u32,
                data1: vk.dag_main_part_index_to_commit_index(part_index) as u32,
                data2: offset as u32,
                constant: F::ZERO,
            },
            Entry::Public => FlatSymbolicConstraintNode {
                kind: NodeKind::VarPublicValue as u8,
                data0: var.index as u32,
                data1: 0,
                data2: 0,
                constant: F::ZERO,
            },
            Entry::Permutation { .. } | Entry::Challenge | Entry::Exposed => unreachable!(),
        },
        SymbolicExpressionNode::IsFirstRow => FlatSymbolicConstraintNode {
            kind: NodeKind::SelIsFirst as u8,
            data0: 0,
            data1: 0,
            data2: 0,
            constant: F::ZERO,
        },
        SymbolicExpressionNode::IsLastRow => FlatSymbolicConstraintNode {
            kind: NodeKind::SelIsLast as u8,
            data0: 0,
            data1: 0,
            data2: 0,
            constant: F::ZERO,
        },
        SymbolicExpressionNode::IsTransition => FlatSymbolicConstraintNode {
            kind: NodeKind::SelIsTransition as u8,
            data0: 0,
            data1: 0,
            data2: 0,
            constant: F::ZERO,
        },
        SymbolicExpressionNode::Constant(val) => FlatSymbolicConstraintNode {
            kind: NodeKind::Constant as u8,
            data0: 0,
            data1: 0,
            data2: 0,
            constant: *val,
        },
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            ..
        } => FlatSymbolicConstraintNode {
            kind: NodeKind::Add as u8,
            data0: *left_idx as u32,
            data1: *right_idx as u32,
            data2: 0,
            constant: F::ZERO,
        },
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            ..
        } => FlatSymbolicConstraintNode {
            kind: NodeKind::Sub as u8,
            data0: *left_idx as u32,
            data1: *right_idx as u32,
            data2: 0,
            constant: F::ZERO,
        },
        SymbolicExpressionNode::Neg { idx, .. } => FlatSymbolicConstraintNode {
            kind: NodeKind::Neg as u8,
            data0: *idx as u32,
            data1: 0,
            data2: 0,
            constant: F::ZERO,
        },
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            ..
        } => FlatSymbolicConstraintNode {
            kind: NodeKind::Mul as u8,
            data0: *left_idx as u32,
            data1: *right_idx as u32,
            data2: 0,
            constant: F::ZERO,
        },
    }
}

pub(super) fn flatten_unused_symbolic_variable(
    unused: &SymbolicVariable<F>,
) -> FlatSymbolicVariable {
    match unused.entry {
        Entry::Preprocessed { offset } => FlatSymbolicVariable {
            entry_kind: NodeKind::VarPreprocessed as u8,
            index: unused.index as u32,
            part_index: 0,
            offset: offset as u32,
        },
        Entry::Main { part_index, offset } => FlatSymbolicVariable {
            entry_kind: NodeKind::VarMain as u8,
            index: unused.index as u32,
            part_index: part_index as u32,
            offset: offset as u32,
        },
        Entry::Public => unreachable!("public variable cannot be unused"),
        Entry::Permutation { .. } | Entry::Challenge | Entry::Exposed => {
            unreachable!("variable type not supported")
        }
    }
}
