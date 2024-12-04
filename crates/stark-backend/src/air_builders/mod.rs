use p3_air::AirBuilder;
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};

pub mod debug;
pub mod prover;
pub mod sub;
pub mod symbolic;
pub mod verifier;

pub type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

/// AIR builder that supports main trace matrix which is partitioned
/// into sub-matrices which belong to different commitments.
pub trait PartitionedAirBuilder: AirBuilder {
    /// Cached main trace matrix.
    fn cached_mains(&self) -> &[Self::M];
    /// Common main trace matrix. Panic if there is no common main trace.
    fn common_main(&self) -> &Self::M;
}
