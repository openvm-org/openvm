use openvm_circuit_primitives_derive::ColsRef;

trait ExampleConfig {
    const N: usize;
}
struct ExampleConfigImplA;
impl ExampleConfig for ExampleConfigImplA {
    const N: usize = 5;
}

#[allow(dead_code)]
#[derive(ColsRef)]
#[config(ExampleConfig)]
struct ExampleCols<T, const N: usize> {
    arr: [T; N],
    // arr: [T; { N }],
    sum: T,
    // primitive: u32,
    // array_of_primitive: [u32; { N }],
}

#[test]
fn debug() {
    let input = [1, 2, 3, 4, 5, 15];
    let test: ExampleColsRef<u32> = ExampleColsRef::from::<ExampleConfigImplA>(&input);
    println!("{}, {}", test.arr, test.sum);
}
