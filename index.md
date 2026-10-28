| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 1,690 |  12,000,265 |  369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 9,660 |  18,655,329 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 5,335 |  14,793,960 |  592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 691 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 433 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 587 |  1,745,757 |  193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-17201d485a80300d698ae460f92bf6e1fe99c5ef.md) | 2,293 |  2,579,903 |  491 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/17201d485a80300d698ae460f92bf6e1fe99c5ef

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40720302296)
