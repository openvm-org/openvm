use super::IsEqualChip;
use afs_stark_backend::interaction::Chip;
use p3_field::Field;

// No interactions
impl<F: Field> Chip<F> for IsEqualChip {}
