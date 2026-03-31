| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/fibonacci-c0d297d73c852c783fecc84a62bd321861a10c24.md) | 3,814 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/keccak-c0d297d73c852c783fecc84a62bd321861a10c24.md) | 18,359 |  18,655,329 |  3,276 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/regex-c0d297d73c852c783fecc84a62bd321861a10c24.md) | 1,420 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/ecrecover-c0d297d73c852c783fecc84a62bd321861a10c24.md) | 651 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/pairing-c0d297d73c852c783fecc84a62bd321861a10c24.md) | 906 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/kitchen_sink-c0d297d73c852c783fecc84a62bd321861a10c24.md) | 2,281 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c0d297d73c852c783fecc84a62bd321861a10c24

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23776551959)
