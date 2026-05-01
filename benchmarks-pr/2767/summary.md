| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/fibonacci-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 3,829 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/keccak-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 18,323 |  18,655,329 |  3,265 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/sha2_bench-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 8,890 |  14,793,960 |  1,382 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/regex-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 1,399 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/ecrecover-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 639 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/pairing-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 899 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/kitchen_sink-ff46a6c730fe1e3972e78e51892d469622fda8a5.md) | 2,086 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ff46a6c730fe1e3972e78e51892d469622fda8a5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25206832485)
