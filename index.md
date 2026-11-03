| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 1,662 |  12,000,265 |  369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 9,587 |  18,655,329 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 5,242 |  14,793,960 |  587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 687 |  4,137,067 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 430 |  123,583 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 584 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-d218a6dd20127bf576f4c986b836b6c81567b327.md) | 2,299 |  2,579,903 |  492 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d218a6dd20127bf576f4c986b836b6c81567b327

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41701286631)
