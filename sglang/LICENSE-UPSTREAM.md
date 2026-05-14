# Upstream attribution — `dusty-nv/jetson-containers`

The `sglang/orin/` recipe in this subtree is vendored from
[`dusty-nv/jetson-containers`](https://github.com/dusty-nv/jetson-containers),
specifically the path `packages/llm/sglang/`.

* **Vendored on:** 2026-05-14
* **Upstream commit at vendoring:** `6ec74990dc4b84f3cbba86c2def7f232db9d0eaf`
* **Upstream license:** MIT
* **Maintainer:** Dustin Franklin (NVIDIA DevRel) and contributors

The license text below is the upstream `LICENSE.md` reproduced
verbatim, as required by the MIT terms. This repository is also
MIT-licensed (see `/LICENSE`); the two are compatible.

When re-syncing from upstream, update the commit SHA above and diff the
verbatim files (`Dockerfile`, `build.sh`, `install.sh`, `test.py`) against
the upstream snapshot at the new SHA. `orin/config.py` is intentionally
divergent (pinned for Orin/JP6/CUDA 12.6 instead of upstream's CUDA-13
pin) and should not be overwritten by a sync.

---

## Upstream license (MIT)

```
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
