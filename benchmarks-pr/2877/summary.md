| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/fibonacci-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 3,993 |  12,000,265 |  1,167 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/keccak-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 22,133 |  18,655,329 |  4,685 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/sha2_bench-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 9,618 |  14,793,960 |  1,846 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/regex-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 1,499 |  4,137,067 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/ecrecover-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 604 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/pairing-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 945 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/kitchen_sink-3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc.md) | 4,155 |  2,579,903 |  888 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3a76cddb3cfd6b2ef00ef159f9c18b72082b4fdc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27422618556)
