| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/fibonacci-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 3,103 |  12,000,265 |  680 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/keccak-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 16,673 |  18,655,329 |  3,103 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/sha2_bench-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 9,082 |  14,793,960 |  1,110 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/regex-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 1,143 |  4,137,067 |  348 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/ecrecover-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 598 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/pairing-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 951 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/kitchen_sink-8d3db63af4933bd36643dbf094e852b36207cf1f.md) | 4,112 |  2,579,903 |  883 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8d3db63af4933bd36643dbf094e852b36207cf1f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27832959585)
