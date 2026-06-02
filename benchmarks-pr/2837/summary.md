| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/fibonacci-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 3,761 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/keccak-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 18,354 |  18,655,329 |  3,330 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/sha2_bench-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 9,883 |  14,793,960 |  1,452 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/regex-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 1,399 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/ecrecover-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 598 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/pairing-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 877 |  1,745,757 |  252 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2837/kitchen_sink-617dd8451b6585f9c44aa399e17ca7cc79ee035a.md) | 3,842 |  2,579,903 |  949 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/617dd8451b6585f9c44aa399e17ca7cc79ee035a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26837610837)
