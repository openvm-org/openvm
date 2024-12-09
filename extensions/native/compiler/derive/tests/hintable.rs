use axvm_native_compiler::prelude::*;
use axvm_native_compiler_derive::Hintable;
use axvm_native_recursion::{hints::InnerVal, types::InnerConfig};
use ax_stark_backend::p3_field::AbstractField;

#[derive(Hintable)]
struct TestStruct {
    a: usize,
    b: usize,
    c: usize,
}

#[test]
fn test_macro() {
    let x = TestStruct { a: 1, b: 2, c: 3 };
    let stream = axvm_native_recursion::hints::Hintable::<InnerConfig>::write(&x);
    assert_eq!(
        stream,
        [1, 2, 3]
            .map(|x| vec![InnerVal::from_canonical_usize(x)])
            .to_vec()
    );
}
