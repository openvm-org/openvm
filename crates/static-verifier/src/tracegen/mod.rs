//! Graph-based witness generator for [`StaticVerifierCircuit`]: records the
//! populate trace once as a dataflow IR ([`ir_builder`]) and interprets it in
//! parallel against a shared tape ([`graph_executor`]) using standalone opcode
//! replay routines ([`opcode_impl`]).
//!
//! [`StaticVerifierCircuit`]: crate::StaticVerifierCircuit

pub mod graph_executor;
pub mod ir_builder;
pub mod opcode_impl;
