| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 959 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 15,664 |  14,365,133 |  3,006 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 8,356 |  11,167,961 |  1,022 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 1,205 |  4,090,656 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 435 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 578 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-f25fa657ef952641bd5dc1983b3ce3cafb5a5af6.md) | 3,822 |  1,979,971 |  861 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f25fa657ef952641bd5dc1983b3ce3cafb5a5af6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28953253713)
