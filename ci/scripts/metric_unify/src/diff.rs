use crate::metric::Metric;

pub fn diff_metrics(news: Vec<Metric>, olds: Vec<Metric>) -> Vec<Metric> {
    let mut results = Vec::with_capacity(news.len());
    for new in news {
        let old = olds
            .iter()
            .find(|old| **old == new && old.value != new.value);
        if let Some(old) = old {
            results.push(Metric {
                diff: Some(new.value - old.value),
                diff_percentage: Some((new.value - old.value) / old.value * 100.0),
                ..new
            });
        } else {
            results.push(new.clone());
        }
    }

    results
}
