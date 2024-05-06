use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};

pub mod debug;
pub mod prover;
pub mod symbolic;
pub mod verifier;

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;
