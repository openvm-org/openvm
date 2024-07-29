use afs_primitives::offline_checker::columns::{OfflineCheckerCols, OfflineCheckerColsMut};

use super::PageOfflineChecker;

#[derive(Debug, derive_new::new)]
pub struct PageOfflineCheckerCols<T> {
    pub offline_checker_cols: OfflineCheckerCols<T>,
    /// this bit indicates if this row comes from the initial page
    pub is_initial: T,
    /// this bit indicates if this is the final row of an idx and that it should be sent to the final chip
    pub is_final_write: T,
    /// this bit indicates if this is the final row of an idx and that it that it was deleted (shouldn't be sent to the final chip)
    pub is_final_delete: T,

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
            is_read: slc[offline_checker_cols_width + 3].clone(),
            is_write: slc[offline_checker_cols_width + 4].clone(),
            is_delete: slc[offline_checker_cols_width + 5].clone(),
        }
    }

    pub fn width(oc: &PageOfflineChecker) -> usize {
        oc.offline_checker.air_width() + 6
    }
}

pub struct PageOfflineCheckerColsMut<'a, T> {
    pub offline_checker_cols: OfflineCheckerColsMut<'a, T>,
    /// this bit indicates if this row comes from the initial page
    pub is_initial: &'a mut T,
    /// this bit indicates if this is the final row of an idx and that it should be sent to the final chip
    pub is_final_write: &'a mut T,
    /// this bit indicates if this is the final row of an idx and that it that it was deleted (shouldn't be sent to the final chip)
    pub is_final_delete: &'a mut T,

    /// 1 if the operation is a read, 0 otherwise
    pub is_read: &'a mut T,
    /// 1 if the operation is a write, 0 otherwise
    pub is_write: &'a mut T,
    /// 1 if the operation is a delete, 0 otherwise
    pub is_delete: &'a mut T,
}

impl<'a, T> PageOfflineCheckerColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], oc: &PageOfflineChecker) -> Self {
        let oc_width = oc.offline_checker.air_width();
        let (oc_cols, rest) = slc.split_at_mut(oc_width);

        let offline_checker_cols = OfflineCheckerColsMut::from_slice(oc_cols, &oc.offline_checker);
        let (is_initial, rest) = rest.split_first_mut().unwrap();
        let (is_final_write, rest) = rest.split_first_mut().unwrap();
        let (is_final_delete, rest) = rest.split_first_mut().unwrap();
        let (is_read, rest) = rest.split_first_mut().unwrap();
        let (is_write, rest) = rest.split_first_mut().unwrap();
        let (is_delete, _) = rest.split_first_mut().unwrap();

        Self {
            offline_checker_cols,
            is_initial,
            is_final_write,
            is_final_delete,
            is_read,
            is_write,
            is_delete,
        }
    }
}
