pub mod output_writer;
pub mod table_gen;
pub mod tracing;

/// Gets the largest power of two less than n
pub fn nearest_power_of_two_floor(n: usize) -> usize {
    let mut i = 1;
    while i < n {
        i *= 2;
    }
    i / 2
}
