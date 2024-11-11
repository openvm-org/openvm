use std::fmt::Display;

pub(crate) fn write_status(style: &dyn Display, status: &str, msg: &str) {
    println!("{style}{status:>12}{style:#} {msg}");
}
