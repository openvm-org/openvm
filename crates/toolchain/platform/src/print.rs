/// Print a UTF-8 string to stdout on host machine for debugging purposes.
#[allow(unused_variables)]
pub fn print<S: AsRef<str>>(s: S) {
    #[cfg(all(not(target_os = "zkvm"), feature = "std"))]
    print!("{}", s.as_ref());
    #[cfg(target_os = "zkvm")]
    openvm_rv32im_guest::print_str_from_bytes(s.as_ref().as_bytes());
}

pub fn println<S: AsRef<str>>(s: S) {
    print(s);
    print("\n");
}
