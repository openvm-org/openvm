| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-6e3de3637fe41a3dcfd89c21e81f2a8305fc363c.md) | 3,844 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-6e3de3637fe41a3dcfd89c21e81f2a8305fc363c.md) | 18,419 |  18,655,329 |  3,284 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-6e3de3637fe41a3dcfd89c21e81f2a8305fc363c.md) | 1,425 |  4,137,067 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-6e3de3637fe41a3dcfd89c21e81f2a8305fc363c.md) | 647 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-6e3de3637fe41a3dcfd89c21e81f2a8305fc363c.md) | 905 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-6e3de3637fe41a3dcfd89c21e81f2a8305fc363c.md) | 2,285 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6e3de3637fe41a3dcfd89c21e81f2a8305fc363c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23955448948)
