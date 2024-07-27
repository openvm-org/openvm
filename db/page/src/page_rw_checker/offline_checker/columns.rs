use afs_primitives::offline_checker::columns::OfflineCheckerCols;

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
    pub fn write_to_buf(self, buf: &mut Vec<T>) {
        self.offline_checker_cols.to_buf(buf);
        buf.push(self.is_initial);
        buf.push(self.is_final_write);
        buf.push(self.is_final_delete);
        buf.push(self.is_read);
        buf.push(self.is_write);
        buf.push(self.is_delete);
    }
    pub fn flatten(self, oc: &PageOfflineChecker) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::width(oc));
        self.write_to_buf(&mut buf);
        buf
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
