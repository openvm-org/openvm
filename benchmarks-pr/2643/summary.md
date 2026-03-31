| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2643/fibonacci-837140b8e34c68712273877b17c13bff99787810.md) | 3,836 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2643/keccak-837140b8e34c68712273877b17c13bff99787810.md) | 18,602 |  18,655,329 |  3,311 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2643/regex-837140b8e34c68712273877b17c13bff99787810.md) | 1,434 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2643/ecrecover-837140b8e34c68712273877b17c13bff99787810.md) | 654 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2643/pairing-837140b8e34c68712273877b17c13bff99787810.md) | 908 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2643/kitchen_sink-837140b8e34c68712273877b17c13bff99787810.md) | 2,272 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/837140b8e34c68712273877b17c13bff99787810

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23822951103)
