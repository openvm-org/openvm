mod add;
mod algorithms;
mod div;
mod log;
mod modular;
mod mul;
mod pow;
mod root;

pub fn group(c: &mut criterion::Criterion) {
    add::group(c);
    mul::group(c);
    div::group(c);
    pow::group(c);
    log::group(c);
    root::group(c);
    modular::group(c);
    algorithms::group(c);
}
