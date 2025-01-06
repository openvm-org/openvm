#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm_native_guest_macro::edsl_kernel! {
    fn function_name(foo: usize | Felt<F>, bar: usize | Felt<F>) -> usize | Felt<F> {
        compiler_output.txt
    }
}


openvm::entry!(main);

fn main() {
    let x = 333;
    let y = 444;
    let z = function_name(x, y);
    if z != 777 {
        panic!();
    }
    //println!("Hello, world!");
}
