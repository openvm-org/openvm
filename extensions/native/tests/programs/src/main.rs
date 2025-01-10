#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm_native_guest_macro::native_kernel! {
    fn function_name(n: usize | Felt<F>) -> usize | Felt<F> {
        compiler_output.txt
    }
}

openvm::entry!(main);

fn main() {
    let answers = [0, 1, 1, 2, 3, 5, 8, 13];
    for (i, answer) in answers.into_iter().enumerate() {
        if function_name(i) != answer {
            panic!();
        }
    }
}
