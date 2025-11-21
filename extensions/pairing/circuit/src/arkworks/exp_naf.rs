use ark_ff::Field;

/// Helper function to implement NAF (Non-Adjacent Form) exponentiation for arkworks fields
///
/// This performs exponentiation using the NAF representation of the exponent,
/// which is more efficient than binary exponentiation for large exponents.
///
/// # Arguments
/// * `base` - The base element to exponentiate
/// * `is_positive` - If false, uses the inverse of the base
/// * `digits_naf` - The NAF representation of the exponent
///
/// # Returns
/// The result of base^exponent where exponent is represented by digits_naf
pub fn exp_naf<F: Field>(base: &F, is_positive: bool, digits_naf: &[i8]) -> F {
    if digits_naf.is_empty() {
        return F::ONE;
    }

    let mut element = *base;
    if !is_positive {
        element = element.inverse().expect("attempted to invert zero element");
    }

    let element_inv = if digits_naf.contains(&-1) {
        Some(
            element
                .inverse()
                .expect("negative digit requires invertible element"),
        )
    } else {
        None
    };

    let mut res = F::ONE;
    for &digit in digits_naf.iter().rev() {
        res = res.square();
        if digit == 1 {
            res *= element;
        } else if digit == -1 {
            res *= element_inv.as_ref().unwrap();
        }
    }
    res
}
