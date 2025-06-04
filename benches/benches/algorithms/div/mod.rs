mod reciprocal;
mod small;

pub fn group(criterion: &mut criterion::Criterion) {
    reciprocal::group(criterion);
    small::group(criterion);
}
