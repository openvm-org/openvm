//! Under construction
use p3_field::Field;

/// The most common adapter for RV32 support.
/// Reads `NUM_READS` register values (each 4 byte cells) and writes `NUM_WRITES` register values.
// can fix NUM_WRITES=1 if easier
pub struct Rv32RegisterAdapter<F: Field, const NUM_READS: usize, const NUM_WRITES: usize> {
    _marker: std::marker::PhantomData<F>,
}

/// Reads `NUM_READS` register values and uses each register value as a pointer to batch read `READ_SIZE` memory cells from
/// address starting at the pointer value.
/// Reads `NUM_WRITES` register values and uses each register value as a pointer to batch write `WRITE_SIZE` memory cells
/// with address starting at the pointer value.
pub struct Rv32HeapAdapter<
    F: Field,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    _marker: std::marker::PhantomData<F>,
}
