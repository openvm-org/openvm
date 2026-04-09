| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-2bff61fe5a66af846d395dd17deca841bdf60f8f.md) | 3,916 |  12,000,265 |  970 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-2bff61fe5a66af846d395dd17deca841bdf60f8f.md) | 18,374 |  18,655,329 |  3,286 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-2bff61fe5a66af846d395dd17deca841bdf60f8f.md) | 1,422 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-2bff61fe5a66af846d395dd17deca841bdf60f8f.md) | 645 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-2bff61fe5a66af846d395dd17deca841bdf60f8f.md) | 913 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-2bff61fe5a66af846d395dd17deca841bdf60f8f.md) | 2,154 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2bff61fe5a66af846d395dd17deca841bdf60f8f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24186846237)
