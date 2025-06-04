mod div;
mod gcd;
mod mul;

pub fn group(criterion: &mut criterion::Criterion) {
    self::mul::group(criterion);
    self::div::group(criterion);
    self::gcd::group(criterion);
}
