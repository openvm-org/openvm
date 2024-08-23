use std::cmp::Ordering;

pub fn compare_vecs(x: Vec<u32>, y: Vec<u32>) -> Ordering {
    for (a, b) in x.iter().zip(y.iter()) {
        match a.cmp(b) {
            Ordering::Less => return Ordering::Less,
            Ordering::Greater => return Ordering::Greater,
            Ordering::Equal => continue,
        }
    }
    Ordering::Equal
}
