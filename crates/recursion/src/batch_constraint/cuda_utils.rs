use openvm_cuda_backend::prelude::{Digest, F};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicExpressionNode,
    },
    keygen::types::{MultiStarkVerifyingKey, StarkVerifyingKey},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::PrimeCharacteristicRing;

use crate::batch_constraint::expr_eval::{build_cached_records, NodeKind};

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
    pub node_idx: u32,
    pub attrs: [u32; 3],
    pub fanout: u32,
    pub is_constraint: bool,
    pub constraint_idx: u32,
}

pub(crate) fn build_cached_gpu_records(
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
) -> Vec<CachedGpuRecord> {
    build_cached_records(child_vk)
        .iter()
        .map(|r| CachedGpuRecord {
            node_idx: r.node_idx as u32,
            attrs: r.attrs.map(|x| x as u32),
            fanout: r.fanout as u32,
            is_constraint: r.is_constraint,
            constraint_idx: r.constraint_idx as u32,
        })
        .collect()
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
