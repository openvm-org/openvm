mod intmodn;
pub use intmodn::*;

/// Adds two IntModN, returns the result in a new IntModN.
#[inline(always)]
pub fn add_mod_n(a: &IntModN, b: &IntModN) -> IntModN {
    #[cfg(not(target_os = "zkvm"))]
    {
        a + b
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}

/// Subtracts two IntModN, returns the result in a new IntModN.
#[inline(always)]
pub fn sub_mod_n(a: &IntModN, b: &IntModN) -> IntModN {
    #[cfg(not(target_os = "zkvm"))]
    {
        a - b
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}

/// Multiplies two IntModN, returns the result in a new IntModN.
#[inline(always)]
pub fn mul_mod_n(a: &IntModN, b: &IntModN) -> IntModN {
    #[cfg(not(target_os = "zkvm"))]
    {
        a * b
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}

/// Divides two IntModN, returns the result in a new IntModN. Division by
/// zero is undefined behaviour.
#[inline(always)]
pub fn div_mod_n(a: &IntModN, b: &IntModN) -> IntModN {
    #[cfg(not(target_os = "zkvm"))]
    {
        a / b
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}
