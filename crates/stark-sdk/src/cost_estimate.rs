use std::{marker::PhantomData, ops::Add};

use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    keygen::types::StarkVerifyingKey,
    p3_field::AbstractExtensionField,
};

use crate::config::FriParameters;

/// Properties of a multi-trace circuit necessary to estimate verifier cost.
#[derive(Clone, Copy, Debug)]
pub struct VerifierCostParameters {
    /// Total number of base field columns across all AIR traces before challenge.
    pub num_main_columns: usize,
    /// Total number of base field columns across all AIR traces for logup permutation.
    pub num_perm_columns: usize,
    /// log_2 Maximum height of an AIR trace.
    pub log_max_height: usize,
    /// Degree of quotient polynomial. This is `max_constraint_degree - 1`.
    pub quotient_degree: usize,
}

/// Mmcs batch verification consist of hashing the leaf and then a normal Merkle proof.
/// We separate the cost of hashing (which requires proper padding to be a crytographic hash) from the cost of
/// 2-to-1 compression function on the hash digest because in tree proofs the internal layers do not need to use
/// a compression function with padding.
///
/// Currently the estimate ignores the additional details of hashing in matrices of different heights.
#[derive(Clone, Copy, Debug)]
pub struct MmcsVerifyBatchCostEstimate {
    /// Hash cost in terms of number of field elments to hash. To convert to true hash cost, it depends on the rate
    /// of the cryptographic hash.
    pub num_f_to_hash: usize,
    /// Number of calls of 2-to-1 compression function.
    pub num_compress: usize,
}

impl MmcsVerifyBatchCostEstimate {
    /// `width` is number of base field columns.
    /// `max_log_height_lde` is the height of the MMCS (which includes blowup)
    pub fn from_dim(width: usize, max_log_height_lde: usize) -> Self {
        Self {
            num_f_to_hash: width,
            num_compress: max_log_height_lde,
        }
    }
}

impl Add for MmcsVerifyBatchCostEstimate {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            num_f_to_hash: self.num_f_to_hash + rhs.num_f_to_hash,
            num_compress: self.num_compress + rhs.num_compress,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FriOpenInputCostEstimate {
    /// Cost from MMCS batch verification.
    pub mmcs: MmcsVerifyBatchCostEstimate,
    /// Number of operations of the form $+ \alpha^? \frac{M_j(\zeta) - y_{ij}}{\zeta - z_i}$ in the reduced opening evaluation.
    pub num_ro_eval: usize,
}

impl FriOpenInputCostEstimate {
    /// `width` is number of base field columns.
    /// `max_log_height` is the trace height, before blowup.
    /// `num_points` is number of points to open.
    pub fn new(
        width: usize,
        max_log_height: usize,
        num_points: usize,
        fri_params: FriParameters,
    ) -> Self {
        let mut mmcs =
            MmcsVerifyBatchCostEstimate::from_dim(width, max_log_height + fri_params.log_blowup);
        mmcs.num_compress *= fri_params.num_queries;
        mmcs.num_f_to_hash *= fri_params.num_queries;
        let num_ro_eval = width * num_points * fri_params.num_queries;
        Self {
            mmcs: MmcsVerifyBatchCostEstimate::from_dim(width, max_log_height),
            num_ro_eval,
        }
    }
}

impl Add for FriOpenInputCostEstimate {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mmcs: self.mmcs + rhs.mmcs,
            num_ro_eval: self.num_ro_eval + rhs.num_ro_eval,
        }
    }
}

pub struct FriQueryCostEstimate {
    /// Cost from MMCS batch verification.
    pub mmcs: MmcsVerifyBatchCostEstimate,
    /// Number of single FRI fold evaluations: `e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0])`.
    pub num_fri_folds: usize,
}

impl FriQueryCostEstimate {
    /// `max_log_height` is the trace height, before blowup.
    pub fn new(max_log_height: usize, fri_params: FriParameters) -> Self {
        let mut mmcs = MmcsVerifyBatchCostEstimate {
            num_f_to_hash: 2 * max_log_height,
            num_compress: max_log_height * (max_log_height + fri_params.log_blowup - 1) / 2,
        };
        mmcs.num_compress *= fri_params.num_queries;
        mmcs.num_f_to_hash *= fri_params.num_queries;
        let num_fri_folds = max_log_height * fri_params.num_queries;
        Self {
            mmcs,
            num_fri_folds,
        }
    }
}

impl Add for FriQueryCostEstimate {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mmcs: self.mmcs + rhs.mmcs,
            num_fri_folds: self.num_fri_folds + rhs.num_fri_folds,
        }
    }
}

pub struct FriVerifierCostEstimate {
    pub open_input: FriOpenInputCostEstimate,
    pub query: FriQueryCostEstimate,
    /// We currently ignore the constraint evaluation cost because it does not scale with number of FRI queries.
    pub constraint_eval: PhantomData<usize>,
}

impl FriVerifierCostEstimate {
    pub fn new(
        params: VerifierCostParameters,
        fri_params: FriParameters,
        ext_degree: usize,
    ) -> Self {
        // Go through different rounds: preprocessed, main, permutation, quotient

        // TODO: ignoring preprocessed trace opening for now

        // Main
        // Currently assumes opening at just zeta, omega * zeta
        let mut open_input = FriOpenInputCostEstimate::new(
            params.num_main_columns,
            params.log_max_height,
            2,
            fri_params,
        );
        let mut query = FriQueryCostEstimate::new(params.log_max_height, fri_params);

        // Permutation
        // Currently assumes opening at just zeta, omega * zeta
        open_input = open_input
            + FriOpenInputCostEstimate::new(
                params.num_perm_columns,
                params.log_max_height,
                2,
                fri_params,
            );
        query = query + FriQueryCostEstimate::new(params.log_max_height, fri_params);

        // Add quotient polynomial opening contribution
        // Quotient only opens at single point zeta
        open_input = open_input
            + FriOpenInputCostEstimate::new(
                params.quotient_degree * ext_degree,
                params.log_max_height,
                1,
                fri_params,
            );
        query = query + FriQueryCostEstimate::new(params.log_max_height, fri_params);

        Self {
            open_input,
            query,
            constraint_eval: PhantomData,
        }
    }

    pub fn from_vk<SC: StarkGenericConfig>(
        vks: &[&StarkVerifyingKey<SC>],
        fri_params: FriParameters,
        log_max_height: usize,
    ) -> Self {
        let num_main_columns: usize = vks
            .iter()
            .map(|vk| {
                vk.params.width.common_main + vk.params.width.cached_mains.iter().sum::<usize>()
            })
            .sum();
        let ext_degree = <SC::Challenge as AbstractExtensionField<Val<SC>>>::D;
        let num_perm_columns: usize = vks
            .iter()
            .map(|vk| vk.params.width.after_challenge.iter().sum::<usize>())
            .sum::<usize>()
            * ext_degree;
        let quotient_degree = vks.iter().map(|vk| vk.quotient_degree).max().unwrap_or(0);
        Self::new(
            VerifierCostParameters {
                num_main_columns,
                num_perm_columns,
                log_max_height,
                quotient_degree,
            },
            fri_params,
            ext_degree,
        )
    }
}
