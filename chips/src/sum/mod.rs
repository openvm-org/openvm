use crate::is_less_than::{IsLessThanAir, IsLessThanChip};

pub mod air;
pub mod bridge;
pub mod columns;
#[cfg(test)]
pub mod tests;
pub mod trace;

/// The `SumAir` defines constraints for a trace matrix that accumulates the sums of
/// values grouped by keys from key-value pair inputs.
///
/// Each state in valid trace matrix is a 4-tuple `(key, value, partial_sum, is_final)`.
/// - `key`: Defines the group.
/// - `value`: The value associated with the key for that row.
/// - `partial_sum`: The cumulative sum of values for the key up to the current row.
/// - `is_final`: Indicates whether this row is the last for the current group (`1` if true, otherwise `0`).
///
/// The data for `key` and `value` is sourced from the input bus. For each unique key, a `(key, sum)` pair
/// is sent to the output bus, where `sum` is the total sum of all values associated with that key.
pub struct SumAir {
    input_bus: usize,
    output_bus: usize,

    is_lt_air: IsLessThanAir,
}

pub struct SumChip {
    air: SumAir,
    is_lt_chip: IsLessThanChip,
}

impl SumChip {
    pub fn new(input_bus: usize, output_bus: usize, is_less_than_chip: IsLessThanChip) -> Self {
        Self {
            air: SumAir {
                input_bus,
                output_bus,
                is_lt_air: is_less_than_chip.air.clone(),
            },
            is_lt_chip: is_less_than_chip,
        }
    }
}
