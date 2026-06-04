| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/fibonacci-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 3,730 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/keccak-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 18,280 |  18,655,329 |  3,331 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/sha2_bench-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 9,986 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/regex-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 1,392 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/ecrecover-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 598 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/pairing-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 882 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/kitchen_sink-25bfc525d4777c6a2d64491d0c0e427fd076c80f.md) | 3,884 |  2,579,903 |  958 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/25bfc525d4777c6a2d64491d0c0e427fd076c80f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26978761107)
