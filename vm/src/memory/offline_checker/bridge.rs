use afs_chips::offline_checker::columns::OfflineCheckerCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_field::PrimeField64;

use super::MemoryOfflineChecker;
use afs_chips::sub_chip::SubAirBridge;

impl<F: PrimeField64> AirBridge<F> for MemoryOfflineChecker {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.offline_checker.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_receive = OfflineCheckerCols::from_slice(&all_cols, &self.offline_checker);
        SubAirBridge::receives(&self.offline_checker, cols_to_receive)
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.offline_checker.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_to_send = OfflineCheckerCols::from_slice(&all_cols, &self.offline_checker);
        SubAirBridge::sends(&self.offline_checker, cols_to_send)
    }
}
