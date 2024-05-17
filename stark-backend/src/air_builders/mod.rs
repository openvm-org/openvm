use p3_air::AirBuilder;
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};

pub mod debug;
pub mod prover;
pub mod symbolic;
pub mod verifier;

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

/// AIR builder that supports main trace matrix which is partitioned
/// into sub-matrices which belong to different commitments.
pub trait PartitionedAirBuilder: AirBuilder {
    /// Main trace matrix, partitioned column-wise into sub-matrices
    fn partitioned_main(&self) -> &[Self::M];
}
