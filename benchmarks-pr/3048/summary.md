| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 470 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 7,253 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 4,751 |  11,167,961 |  536 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 666 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 229 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 324 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb.md) | 2,683 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7ce1c909c7e0dbda0aecb7b0f4afa6c576ff34cb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29951464614)
