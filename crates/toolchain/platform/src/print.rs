/// Print a UTF-8 string to stdout on host machine for debugging purposes.
#[allow(unused_variables)]
pub fn print<S: AsRef<str>>(s: S) {
    #[cfg(all(not(openvm_intrinsics), feature = "std"))]
    print!("{}", s.as_ref());
    #[cfg(openvm_intrinsics)]
    openvm_rv64im_guest::print_str_from_bytes(s.as_ref().as_bytes());
}

pub fn println<S: AsRef<str>>(s: S) {
    print(s);
    print("\n");
}
