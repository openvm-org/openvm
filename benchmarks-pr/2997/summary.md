| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/fibonacci-7a00386544e21601cc544049133ce9d857135d68.md) | 3,045 |  12,000,265 |  683 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/keccak-7a00386544e21601cc544049133ce9d857135d68.md) | 16,458 |  18,655,329 |  3,031 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/sha2_bench-7a00386544e21601cc544049133ce9d857135d68.md) | 9,554 |  14,793,960 |  1,138 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/regex-7a00386544e21601cc544049133ce9d857135d68.md) | 1,204 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/ecrecover-7a00386544e21601cc544049133ce9d857135d68.md) | 512 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/pairing-7a00386544e21601cc544049133ce9d857135d68.md) | 845 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/kitchen_sink-7a00386544e21601cc544049133ce9d857135d68.md) | 4,503 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a00386544e21601cc544049133ce9d857135d68

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29063194512)
