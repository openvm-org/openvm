| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 415 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 8,355 |  14,365,133 |  1,512 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 4,132 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 499 |  4,090,656 |  187 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 220 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 276 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-919347b77e6f398460527acf891dd1a7c07e2d56.md) | 2,009 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/919347b77e6f398460527acf891dd1a7c07e2d56

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29457343771)
