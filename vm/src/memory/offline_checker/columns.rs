use afs_chips::{
    is_equal_vec::columns::IsEqualVecAuxCols, offline_checker::columns::OfflineCheckerCols,
};

use super::MemoryOfflineChecker;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, derive_new::new)]
pub struct MemoryOfflineCheckerCols<T> {
    pub offline_checker_cols: OfflineCheckerCols<T>,
    /// this bit indicates if the data matches the one in the previous row (should be 0 in first row)
    pub same_data: T,
    /// this bit indicates if the idx and data match the one in the previous row (should be 0 in first row)
    /// this is used to reduce the degree of a constraint
    pub same_idx_and_data: T,
    /// auxiliary columns used for same_data
    pub is_equal_data_aux: IsEqualVecAuxCols<T>,
}

impl<T> MemoryOfflineCheckerCols<T>
where
    T: Clone,
{
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.offline_checker_cols.flatten();

        flattened.extend(vec![self.same_data.clone(), self.same_idx_and_data.clone()]);
        flattened.extend(self.is_equal_data_aux.flatten());

        flattened
    }

    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());

        let offline_checker_cols_width = oc.offline_checker.air_width();
        let offline_checker_cols =
            OfflineCheckerCols::from_slice(&slc[..offline_checker_cols_width], &oc.offline_checker);

        Self {
            offline_checker_cols,
            same_data: slc[offline_checker_cols_width].clone(),
            same_idx_and_data: slc[offline_checker_cols_width + 1].clone(),
            is_equal_data_aux: IsEqualVecAuxCols::from_slice(
                &slc[offline_checker_cols_width + 2..],
                oc.offline_checker.data_len,
            ),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        oc.offline_checker.air_width()
            + 2
            + IsEqualVecAuxCols::<T>::get_width(oc.offline_checker.data_len)
    }
}
