| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10.md) | 3,801 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10.md) | 18,540 |  18,655,329 |  3,321 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10.md) | 1,428 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10.md) | 641 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10.md) | 924 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10.md) | 2,157 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/41fac2ae8de225f3fd5f12e42c9c2f7e5edf2a10

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24210160862)
