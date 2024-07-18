use crate::offline_checker::columns::GeneralOfflineCheckerCols;

use super::OfflineChecker;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, derive_new::new)]
pub struct OfflineCheckerCols<T> {
    pub general_cols: GeneralOfflineCheckerCols<T>,
    /// this bit indicates if this row comes from the initial page
    pub is_initial: T,
    /// this bit indicates if this is the final row of an idx and that it should be sent to the final chip
    pub is_final_write: T,
    /// this bit indicates if this is the final row of an idx and that it that it was deleted (shouldn't be sent to the final chip)
    pub is_final_delete: T,
    /// this bit indicates if this row refers to an internal operation
    pub is_internal: T,

    /// this is just is_final_write * 3 (used for interactions)
    pub is_final_write_x3: T,

    /// 1 if the operation is a read, 0 otherwise
    pub is_read: T,
    /// 1 if the operation is a write, 0 otherwise
    pub is_write: T,
    /// 1 if the operation is a delete, 0 otherwise
    pub is_delete: T,
}

impl<T> OfflineCheckerCols<T>
where
    T: Clone,
{
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.general_cols.flatten();

        flattened.extend(vec![
            self.is_initial.clone(),
            self.is_final_write.clone(),
            self.is_final_delete.clone(),
            self.is_internal.clone(),
            self.is_final_write_x3.clone(),
        ]);
        flattened.extend(vec![
            self.is_read.clone(),
            self.is_write.clone(),
            self.is_delete.clone(),
        ]);

        flattened
    }

    pub fn from_slice(slc: &[T], oc: &OfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());

        let general_cols_width = oc.general_offline_checker.air_width();
        let general_cols = GeneralOfflineCheckerCols::from_slice(
            &slc[..general_cols_width],
            &oc.general_offline_checker,
        );

        Self {
            general_cols,
            is_initial: slc[general_cols_width].clone(),
            is_final_write: slc[general_cols_width + 1].clone(),
            is_final_delete: slc[general_cols_width + 2].clone(),
            is_internal: slc[general_cols_width + 3].clone(),
            is_final_write_x3: slc[general_cols_width + 4].clone(),
            is_read: slc[general_cols_width + 5].clone(),
            is_write: slc[general_cols_width + 6].clone(),
            is_delete: slc[general_cols_width + 7].clone(),
        }
    }

    pub fn width(oc: &OfflineChecker) -> usize {
        oc.general_offline_checker.air_width() + 8
    }
}
