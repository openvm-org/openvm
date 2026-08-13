| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 443 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 7,168 |  14,365,133 |  1,595 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 4,125 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 705 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 211 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 236 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-a0cc50c869e4f2fa00f780a7e329d11935cd7846.md) | 2,164 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a0cc50c869e4f2fa00f780a7e329d11935cd7846

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31720214457)
