#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm_native_guest_macro::edsl_kernel! {
    fn function_name(n: usize | Felt<F>) -> usize | Felt<F> {
        compiler_output.txt
    }
}

openvm::entry!(main);

const N: usize = 8;

fn main() {
    let answers: [usize; N] = [0, 1, 1, 2, 3, 5, 8, 13];
    for i in 0..N {
        if function_name(i) != answers[i] {
            panic!();
        }
    }
}