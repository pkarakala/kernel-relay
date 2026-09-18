# Public artifact normalization

The files in this directory are public, path-normalized copies of the recorded Colab outputs. The only edits to the three full JSON traces and three terminal logs replaced an old absolute Colab extraction root with `/content/kernel-relay`. The paired confirmation received the same path replacement and its `source_trace_sha256` was updated to reference the **public** first-run trace. No latency, correctness, reward, source-code, hardware, or isolation values were changed. Original files remain in a private research archive.

| File | Original SHA-256 | Public SHA-256 |
| --- | --- | --- |
| `v2-full-t4-fp16.json` | `a60046f087d4e47dceb843dfc9ca51a1b2a56376561324649ef7ce9625721961` | `7350220764d2bf2accb1621528c7b93121d196473bb74bff14b369e5c55ae5eb` |
| `v2-full-t4-fp16.log` | `191f48688b8c9843fb8621b6b7f5752d5c83a74be662d43e31a07cb964d70028` | `c8228f62ebfc63b3b2c0b506fa459a05786d71644d293c32673641478e079121` |
| `v2-full-t4-fp16-repeat2.json` | `250adea43d31045653021d2a20f119d4b92912f8a0c315208cd6a917b78137d0` | `179d17b3a30d6d0b24847659794096592489780be645583ffcc79b29af98a1ab` |
| `v2-full-t4-fp16-repeat2.log` | `6cb01abaf441cfce26a99f350bf8120ff64e0ec208101fa9342bbdf9621934a2` | `f3c1a6b2288462d1446b3326b021c2b4465a3ec50dcae0f94a1979dec6603a57` |
| `v2-full-t4-fp16-repeat3.json` | `8783542ab93a9835e8181bdf7e3e3c739383c67b434611be2a3c3a9438fa05c1` | `590ae158d677130d2e243db73a40b386e63a978ba7e5df0dba2cdf7da84e795a` |
| `v2-full-t4-fp16-repeat3.log` | `0a3ff2947b9d2b13a8cd2e49faef5348525418e0b9c318293ff34babb86f1edd` | `41f0076ea370fc22a6446e054f832b3fbd853e85f8a6ddc8be0aafa5f1726201` |
| `v2-paired-t4-fp16.json` | `0c42e8a8bcf3d8362e850805db9687c7ef6c87edf925f4ee03ed643a0131ed3b` | `1ef916419d414f0f6c4f0094571ca630684adfdf2307124ccf5acb70fae88ec5` |

The original hashes are disclosed to make the transformation auditable without exposing the old extraction path in this repository. A fresh evaluation under the public project name can produce byte-original public traces in the future; it must not be substituted for these measurements without being run on a GPU.
