| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/fibonacci-b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f.md) | 3,867 |  12,000,265 |  970 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/keccak-b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f.md) | 15,721 |  1,235,218 |  2,209 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/regex-b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f.md) | 1,427 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/ecrecover-b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f.md) | 636 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/pairing-b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f.md) | 910 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/kitchen_sink-b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f.md) | 2,408 |  154,763 |  426 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b6a92877ca4b1b033cc43fe76cd1bc97cedfa60f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23865031151)
