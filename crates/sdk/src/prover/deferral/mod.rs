// Deferral proving has three nested layers:
// - `single_circuit`: proves one concrete deferral circuit and wraps its proofs into deferral
//   leaf/internal-for-leaf proofs.
// - `multi_circuit`: owns the configured set of deferral circuits and produces hook proofs for each
//   present deferral input.
// - `agg`: aggregates hook proofs into the deferral proof consumed by the main SDK prover.
//
// `merkle` is separate because it is used when attaching deferral Merkle proofs to VM STARK
// outputs, after the deferral proving layers have run.
mod agg;
mod merkle;
mod multi_circuit;
mod single_circuit;

pub use agg::*;
pub use merkle::*;
pub use multi_circuit::*;
pub use single_circuit::*;
