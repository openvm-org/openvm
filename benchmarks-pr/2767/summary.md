| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/fibonacci-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 3,765 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/keccak-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 18,473 |  18,655,329 |  3,296 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/sha2_bench-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 8,985 |  14,793,960 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/regex-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 1,424 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/ecrecover-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 638 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/pairing-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 898 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/kitchen_sink-a2b8f933cad4a019a66be2b0f7990a202cbf3dd8.md) | 2,083 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a2b8f933cad4a019a66be2b0f7990a202cbf3dd8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25207124918)
