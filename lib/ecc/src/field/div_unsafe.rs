pub trait DivUnsafe: Sized {
    type Output;

    fn div_unsafe(self, other: &Self) -> Self::Output;
}

pub trait DivAssignUnsafe: Sized {
    fn div_assign_unsafe(&mut self, other: &Self);
}
