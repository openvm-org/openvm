| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-796458ee5da6795eb241854ca948f921d258eb45.md) | 1,656 |  4,000,051 |  529 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-796458ee5da6795eb241854ca948f921d258eb45.md) | 16,403 |  14,365,133 |  3,069 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-796458ee5da6795eb241854ca948f921d258eb45.md) | 10,561 |  11,167,961 |  1,983 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-796458ee5da6795eb241854ca948f921d258eb45.md) | 1,561 |  4,090,656 |  439 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-796458ee5da6795eb241854ca948f921d258eb45.md) | 484 |  112,210 |  314 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-796458ee5da6795eb241854ca948f921d258eb45.md) | 618 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-796458ee5da6795eb241854ca948f921d258eb45.md) | 3,933 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/796458ee5da6795eb241854ca948f921d258eb45

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27367608192)
