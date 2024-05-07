use p3_air::BaseAir;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::IntoParallelIterator;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::air_builders::debug::DebugConstraintBuilder;
use crate::rap::Rap;

/// Check that all constraints vanish on the subgroup.
pub fn check_constraints<A, SC>(
    rap: &A,
    preprocessed: &Option<RowMajorMatrixView<Val<SC>>>,
    main: &RowMajorMatrixView<Val<SC>>,
    perm: &Option<RowMajorMatrixView<SC::Challenge>>,
    perm_challenges: &[SC::Challenge],
    cumulative_sum: Option<SC::Challenge>,
    public_values: &[Val<SC>],
) where
    A: for<'a> Rap<DebugConstraintBuilder<'a, SC>> + BaseAir<Val<SC>> + ?Sized,
    SC: StarkGenericConfig,
{
    let height = main.height();
    if let Some(perm) = perm {
        assert_eq!(height, perm.height());
    }

    // Check that constraints are satisfied.
    (0..height).into_par_iter().for_each(|i| {
        let i_next = (i + 1) % height;

        let (preprocessed_local, preprocessed_next) = preprocessed
            .as_ref()
            .map(|preprocessed| {
                (
                    preprocessed.row_slice(i).to_vec(),
                    preprocessed.row_slice(i_next).to_vec(),
                )
            })
            .unwrap_or((vec![], vec![]));

        let (main_local, main_next) = (&*main.row_slice(i), &*main.row_slice(i_next));

        let (perm_local, perm_next) = perm
            .as_ref()
            .map(|perm| (perm.row_slice(i).to_vec(), perm.row_slice(i_next).to_vec()))
            .unwrap_or((vec![], vec![]));
        let perm_exposed_values = cumulative_sum.map(|s| vec![s]).unwrap_or_default();

        let mut builder = DebugConstraintBuilder {
            row_index: i,
            preprocessed: VerticalPair::new(
                RowMajorMatrixView::new_row(preprocessed_local.as_slice()),
                RowMajorMatrixView::new_row(preprocessed_next.as_slice()),
            ),
            main: VerticalPair::new(
                RowMajorMatrixView::new_row(main_local),
                RowMajorMatrixView::new_row(main_next),
            ),
            perm: VerticalPair::new(
                RowMajorMatrixView::new_row(perm_local.as_slice()),
                RowMajorMatrixView::new_row(perm_next.as_slice()),
            ),
            perm_challenges,
            public_values,
            perm_exposed_values: perm_exposed_values.as_slice(),
            is_first_row: Val::<SC>::zero(),
            is_last_row: Val::<SC>::zero(),
            is_transition: Val::<SC>::one(),
        };
        if i == 0 {
            builder.is_first_row = Val::<SC>::one();
        }
        if i == height - 1 {
            builder.is_last_row = Val::<SC>::one();
            builder.is_transition = Val::<SC>::zero();
        }

        rap.eval(&mut builder);
    });
}
