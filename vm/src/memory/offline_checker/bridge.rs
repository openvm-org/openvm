use afs_chips::offline_checker::columns::GeneralOfflineCheckerCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_field::PrimeField64;

use super::OfflineChecker;
use afs_chips::sub_chip::SubAirBridge;

impl<F: PrimeField64> AirBridge<F> for OfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.general_offline_checker.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive =
            GeneralOfflineCheckerCols::from_slice(&all_cols, &self.general_offline_checker);
        SubAirBridge::receives(&self.general_offline_checker, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.general_offline_checker.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send =
            GeneralOfflineCheckerCols::from_slice(&all_cols, &self.general_offline_checker);
        SubAirBridge::sends(&self.general_offline_checker, cols_to_send)
    }
}
