| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 1,026 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 16,155 |  14,365,133 |  3,032 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 7,993 |  11,167,961 |  984 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 1,164 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 437 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 587 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-aa67344f63edab9fde1dd3c6b5abe2c43532bf72.md) | 3,871 |  1,979,971 |  859 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aa67344f63edab9fde1dd3c6b5abe2c43532bf72

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28190327160)
