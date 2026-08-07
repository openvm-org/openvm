| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 886 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 8,572 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 4,235 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 738 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 499 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 476 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-9f5b102638ab9ba35ce750e0e77fbde2d8999069.md) | 2,355 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9f5b102638ab9ba35ce750e0e77fbde2d8999069

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31201049469)
