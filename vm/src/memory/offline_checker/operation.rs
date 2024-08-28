use derive_new::new;
use p3_air::AirBuilder;

#[derive(Clone, Debug, PartialEq, Eq, new)]
pub struct MemoryOperation<const N: usize, T> {
    pub addr_space: T,
    pub pointer: T,
    pub timestamp: T,
    pub data: [T; N],
    pub enabled: T,
}

impl<const WORD_SIZE: usize, T> MemoryOperation<WORD_SIZE, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> MemoryOperation<WORD_SIZE, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        MemoryOperation {
            addr_space: self.addr_space.into(),
            pointer: self.pointer.into(),
            timestamp: self.timestamp.into(),
            data: self.data.map(Into::into),
            enabled: self.enabled.into(),
        }
    }
}
