// TODO: Ideally upstream PrimeField implements From<T>
pub trait FieldFrom<T> {
    fn from_val(value: T) -> Self;
}

#[cfg(feature = "test-traits")]
use p3_baby_bear::BabyBear;
#[cfg(feature = "test-traits")]
use p3_field::AbstractField;

#[cfg(feature = "test-traits")]
impl FieldFrom<u8> for BabyBear {
    fn from_val(value: u8) -> Self {
        BabyBear::from_canonical_u8(value)
    }
}

#[cfg(feature = "test-traits")]
impl FieldFrom<BabyBear> for BabyBear {
    fn from_val(value: BabyBear) -> Self {
        value
    }
}
