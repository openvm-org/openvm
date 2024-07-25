use afs_primitives::offline_checker::columns::OfflineCheckerCols;

use super::PageOfflineChecker;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, derive_new::new)]
pub struct PageOfflineCheckerCols<T> {
    pub offline_checker_cols: OfflineCheckerCols<T>,
    /// this bit indicates if this row comes from the initial page
    pub is_initial: T,
    /// this bit indicates if this is the final row of an idx and that it should be sent to the final chip
    pub is_final_write: T,
    /// this bit indicates if this is the final row of an idx and that it that it was deleted (shouldn't be sent to the final chip)
    pub is_final_delete: T,

    /// this is just is_final_write * 3 (used for interactions)
    pub is_final_write_x3: T,

    /// 1 if the operation is a read, 0 otherwise
    pub is_read: T,
    /// 1 if the operation is a write, 0 otherwise
    pub is_write: T,
    /// 1 if the operation is a delete, 0 otherwise
    pub is_delete: T,
}

impl<T> PageOfflineCheckerCols<T>
where
    T: Clone,
{
    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.offline_checker_cols.flatten();

        flattened.extend(vec![
            self.is_initial.clone(),
            self.is_final_write.clone(),
            self.is_final_delete.clone(),
            self.is_final_write_x3.clone(),
        ]);
        flattened.extend(vec![
            self.is_read.clone(),
            self.is_write.clone(),
            self.is_delete.clone(),
        ]);

        flattened
    }

    pub fn from_slice(slc: &[T], oc: &PageOfflineChecker) -> Self {
        assert!(slc.len() == oc.air_width());

        let offline_checker_cols_width = oc.offline_checker.air_width();
        let offline_checker_cols =
            OfflineCheckerCols::from_slice(&slc[..offline_checker_cols_width], &oc.offline_checker);

        Self {
            offline_checker_cols,
            is_initial: slc[offline_checker_cols_width].clone(),
            is_final_write: slc[offline_checker_cols_width + 1].clone(),
            is_final_delete: slc[offline_checker_cols_width + 2].clone(),
            is_final_write_x3: slc[offline_checker_cols_width + 3].clone(),
            is_read: slc[offline_checker_cols_width + 4].clone(),
            is_write: slc[offline_checker_cols_width + 5].clone(),
            is_delete: slc[offline_checker_cols_width + 6].clone(),
        }
    }

    pub fn width(oc: &PageOfflineChecker) -> usize {
        oc.offline_checker.air_width() + 7
    }
}
