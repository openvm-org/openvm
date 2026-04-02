| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-76d87cc628bc9a887cf995b215d3c5b28ec54b4f.md) | 3,832 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-76d87cc628bc9a887cf995b215d3c5b28ec54b4f.md) | 18,513 |  18,655,329 |  3,302 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-76d87cc628bc9a887cf995b215d3c5b28ec54b4f.md) | 1,413 |  4,137,067 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-76d87cc628bc9a887cf995b215d3c5b28ec54b4f.md) | 642 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-76d87cc628bc9a887cf995b215d3c5b28ec54b4f.md) | 904 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-76d87cc628bc9a887cf995b215d3c5b28ec54b4f.md) | 2,285 |  2,579,903 |  444 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/76d87cc628bc9a887cf995b215d3c5b28ec54b4f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23898067138)
