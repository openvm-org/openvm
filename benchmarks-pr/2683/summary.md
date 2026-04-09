| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/fibonacci-5a1ac53ad5182f82cad4b6a0473cea48713b327a.md) | 3,789 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/keccak-5a1ac53ad5182f82cad4b6a0473cea48713b327a.md) | 18,570 |  18,655,329 |  3,340 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/regex-5a1ac53ad5182f82cad4b6a0473cea48713b327a.md) | 1,420 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/ecrecover-5a1ac53ad5182f82cad4b6a0473cea48713b327a.md) | 646 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/pairing-5a1ac53ad5182f82cad4b6a0473cea48713b327a.md) | 910 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/kitchen_sink-5a1ac53ad5182f82cad4b6a0473cea48713b327a.md) | 2,146 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5a1ac53ad5182f82cad4b6a0473cea48713b327a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24204921629)
