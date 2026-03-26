| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-60492c7591935ba206419e3b76d582d4c2ecf94d.md) | 3,851 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-60492c7591935ba206419e3b76d582d4c2ecf94d.md) | 15,767 |  1,235,218 |  2,193 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-60492c7591935ba206419e3b76d582d4c2ecf94d.md) | 1,418 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-60492c7591935ba206419e3b76d582d4c2ecf94d.md) | 646 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-60492c7591935ba206419e3b76d582d4c2ecf94d.md) | 929 |  1,745,757 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-60492c7591935ba206419e3b76d582d4c2ecf94d.md) | 2,372 |  154,763 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/60492c7591935ba206419e3b76d582d4c2ecf94d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23616095712)
