| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/fibonacci-139e60819d393c68a1070e3f3987355213917722.md) | 3,692 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/keccak-139e60819d393c68a1070e3f3987355213917722.md) | 18,209 |  18,655,329 |  3,294 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/sha2_bench-139e60819d393c68a1070e3f3987355213917722.md) | 10,080 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/regex-139e60819d393c68a1070e3f3987355213917722.md) | 1,401 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/ecrecover-139e60819d393c68a1070e3f3987355213917722.md) | 601 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/pairing-139e60819d393c68a1070e3f3987355213917722.md) | 890 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2859/kitchen_sink-139e60819d393c68a1070e3f3987355213917722.md) | 3,843 |  2,579,903 |  947 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/139e60819d393c68a1070e3f3987355213917722

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27156504751)
