| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/fibonacci-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 3,830 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/keccak-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 18,674 |  18,655,329 |  3,323 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/sha2_bench-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 9,250 |  14,793,960 |  1,442 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/regex-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 1,426 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/ecrecover-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 648 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/pairing-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 901 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2709/kitchen_sink-0072270bdd1770941fc4e5c730b6c69cc73b78f9.md) | 2,091 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0072270bdd1770941fc4e5c730b6c69cc73b78f9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24477884512)
