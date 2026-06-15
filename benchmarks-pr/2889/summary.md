| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 3,058 |  12,000,265 |  667 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 16,278 |  18,655,329 |  3,013 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 9,223 |  14,793,960 |  1,129 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 1,169 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 599 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 936 |  1,745,757 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-638b3d11143c609db6251cc05ce44f912e39bfc4.md) | 4,099 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/638b3d11143c609db6251cc05ce44f912e39bfc4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27567974655)
