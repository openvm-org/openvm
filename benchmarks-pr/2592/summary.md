| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-3980186a103ac631e6c4a12b67273fdd9c3295ef.md) | 3,790 |  12,000,265 |  932 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-3980186a103ac631e6c4a12b67273fdd9c3295ef.md) | 18,543 |  18,655,329 |  3,311 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-3980186a103ac631e6c4a12b67273fdd9c3295ef.md) | 1,430 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-3980186a103ac631e6c4a12b67273fdd9c3295ef.md) | 664 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-3980186a103ac631e6c4a12b67273fdd9c3295ef.md) | 908 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-3980186a103ac631e6c4a12b67273fdd9c3295ef.md) | 2,288 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3980186a103ac631e6c4a12b67273fdd9c3295ef

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23623530468)
