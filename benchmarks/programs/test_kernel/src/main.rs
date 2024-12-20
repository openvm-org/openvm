#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

/*openvm_edsl_kernel::edsl_kernel! {
    fn function_name(foo: usize | Felt<F>, bar: usize | Felt<F>) -> usize | Felt<F> {
        return builder.eval(foo + bar);
    }
}*/

fn function_name(var_foo: usize, var_bar: usize, ) -> usize {
    let result: usize;
    unsafe {
        core::arch::asm!(
        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_foo}, x0, 3",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 50, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 0, 0, x2, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_foo}, x0, 2",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 50, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 0, 0, x2, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_foo}, x0, 1",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 50, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 0, 0, x2, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_foo}, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_bar}, x0, 3",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 50, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 0, 0, x2, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_bar}, x0, 2",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 50, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 0, 0, x2, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_bar}, x0, 1",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 50, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 0, 0, x2, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 6, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {var_bar}, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 7, 0, x0, x0, 0",
        ".insn i 1, 0, x2, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 7, 0, x0, x0, 0",
        ".insn i 1, 0, x2, x0, 0",
        ".insn i 0, 0, x0, x0, 16",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 124, 7, x31, x31, 15",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 7, 0, x0, x0, 0",
        ".insn i 21, 0, x2, x0, 0",
        ".insn i 120, 7, x31, x31, 15",
        ".insn i 8, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 7, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 7, 0, x0, x0, 0",
        ".insn i 48, 0, x2, x0, 0",
        ".insn i 52, 7, x31, x31, 15",
        ".insn i 60, 7, x31, x31, 15",
        ".insn i 53, 7, x31, x31, 15",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 0, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 5, 0, x0, x0, 0",
        ".insn i 37, 0, x2, x0, 0",
        ".insn i 116, 0, x0, x0, -2048",
        ".insn i 0, 0, {result}, x0, 0",
        ".insn i 52, 7, x31, x31, 15",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 1, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",

        ".insn i 107, 0, x8, x25, -1939",
        ".insn i 4, 0, x0, x0, 0",
        ".insn i 21, 0, x2, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 88, 0, x5, x0, 0",
        ".insn i 0, 0, x0, x0, 0",
        ".insn i 5, 0, x0, x0, 0",

        ".insn i 25, 7, x16, x27, -1941",
        ".insn i 53, 0, x1, x0, 0",

        var_foo = in(reg) var_foo,
        var_bar = in(reg) var_bar,
        result = out(reg) result,
        )
    }
    result
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
