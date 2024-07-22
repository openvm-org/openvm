use afs_derive::AlignedBorrow;
use p3_air::Air;
use p3_air::{AirBuilder, VirtualPairCol};
use p3_field::Field;
use p3_field::PrimeField32;

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

pub fn to_vcols<F: Field>(cols: &[usize]) -> Vec<VirtualPairCol<F>> {
    cols.iter()
        .copied()
        .map(VirtualPairCol::single_main)
        .collect()
}

pub fn and<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    a * b
}

/// Assumes that a and b are boolean
pub fn or<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    a.clone() + b.clone() - and::<AB>(a, b)
}

/// Assumes that a and b are boolean
pub fn implies<AB: AirBuilder>(a: AB::Expr, b: AB::Expr) -> AB::Expr {
    or::<AB>(AB::Expr::one() - a, b)
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
pub struct Word32<T>(pub [T; 2]);
impl<AB, T> Word32<T>
where
    AB: AirBuilder,
    T: AB::Var,
{
    pub fn get_value(&self) -> AB::Expr {
        let upper = AB::Expr::from(self.0[0]);
        let lower = AB::Expr::from(self.0[1]);
        lower + (upper * AB::Expr::from_canonical_u64(1 << 16))
    }
}

// impl<T> Word32<T> {
//     pub fn get_value<AB: AirBuilder<Var = T>>(&self) -> AB::Expr {
//         let lower = AB::Expr::from(self.0[1]);
//         let upper = AB::Expr::from(self.0[0]);
//         lower + (upper * AB::Expr::from_canonical_u64(1 << 16))
//         // AB::Expr::from_canonical_u16(0)
//     }
// }

// impl<AB: AirBuilder> Word32<AB::Var> {
//     fn concat_u8(x: u8, y: u8) -> u16 {
//         ((x as u16) << 8) | (y as u16)
//     }
//     // TODO: verify from and to 32. probably wrong now. Just make it compile first.
//     pub fn from_u32(x: u32) -> Self {
//         let bytes = x.to_le_bytes(); // TODO: should this be le or be?
//         Word32([
//             T::from_canonical_u16(Self::concat_u8(bytes[0], bytes[1])),
//             T::from_canonical_u16(Self::concat_u8(bytes[2], bytes[3])),
//         ])
//     }
// }
