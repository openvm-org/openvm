use axvm_macros::axvm;

#[test]
fn main() {
    axvm!(z = addmod::<777>(&x, &y););
}
