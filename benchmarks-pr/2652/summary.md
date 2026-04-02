| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2652/fibonacci-238ff0754aefca31855d91381e7e332da39fe402.md) | 3,827 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2652/keccak-238ff0754aefca31855d91381e7e332da39fe402.md) | 15,643 |  1,235,218 |  2,192 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2652/regex-238ff0754aefca31855d91381e7e332da39fe402.md) | 1,403 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2652/ecrecover-238ff0754aefca31855d91381e7e332da39fe402.md) | 638 |  122,348 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2652/pairing-238ff0754aefca31855d91381e7e332da39fe402.md) | 913 |  1,745,757 |  277 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2652/kitchen_sink-238ff0754aefca31855d91381e7e332da39fe402.md) | 2,378 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/238ff0754aefca31855d91381e7e332da39fe402

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23904074584)
