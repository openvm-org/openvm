use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Additive hints for control-flow graph construction.
///
/// Producers may derive these facts from LLVM IR, DWARF, symbol tables, or
/// other build-time information. Consumers still validate every PC against the
/// decoded executable.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CfgHints {
    /// Decoded instruction PCs that should begin a block.
    pub basic_block_starts: BTreeSet<u64>,
    /// Source-less targets that also participate in unresolved-jump fallback analysis.
    pub potential_targets: BTreeSet<u64>,
    /// Decoded indirect jump or call PC to its possible decoded target PCs.
    pub indirect_targets: BTreeMap<u64, BTreeSet<u64>>,
}
