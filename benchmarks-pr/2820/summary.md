| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/fibonacci-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 3,763 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/keccak-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 18,774 |  18,655,329 |  3,315 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/sha2_bench-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 10,164 |  14,793,960 |  1,462 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/regex-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 1,395 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/ecrecover-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 602 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/pairing-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 901 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/kitchen_sink-c4a21f514791004a5c3d947b8623d01fdd99331e.md) | 1,890 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c4a21f514791004a5c3d947b8623d01fdd99331e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26537210954)
