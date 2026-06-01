| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-fbb235226df4003e104cc696129add5f09de257e.md) | 3,768 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-fbb235226df4003e104cc696129add5f09de257e.md) | 18,886 |  18,655,329 |  3,343 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-fbb235226df4003e104cc696129add5f09de257e.md) | 10,355 |  14,793,960 |  1,483 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-fbb235226df4003e104cc696129add5f09de257e.md) | 1,400 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-fbb235226df4003e104cc696129add5f09de257e.md) | 608 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-fbb235226df4003e104cc696129add5f09de257e.md) | 882 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-fbb235226df4003e104cc696129add5f09de257e.md) | 1,906 |  2,579,903 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fbb235226df4003e104cc696129add5f09de257e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26782017124)
