| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2628/fibonacci-5a9f9a294a50a974b668d07b82420cd2852d97ed.md) | 3,790 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2628/keccak-5a9f9a294a50a974b668d07b82420cd2852d97ed.md) | 15,797 |  1,235,218 |  2,208 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2628/regex-5a9f9a294a50a974b668d07b82420cd2852d97ed.md) | 1,420 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2628/ecrecover-5a9f9a294a50a974b668d07b82420cd2852d97ed.md) | 633 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2628/pairing-5a9f9a294a50a974b668d07b82420cd2852d97ed.md) | 923 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2628/kitchen_sink-5a9f9a294a50a974b668d07b82420cd2852d97ed.md) | 2,369 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5a9f9a294a50a974b668d07b82420cd2852d97ed

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23753031312)
