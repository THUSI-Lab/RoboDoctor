# Third-party notices

## Direct and hybrid control

Parts of the direct and hybrid implementation are adapted from
[GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy).
The retained source is covered by this copyright and permission notice:

MIT License

Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PrismWAM

The inference implementation in `third_party/prismwam/src/prismwam/` is from
[THUSI-Lab/PrismWAM](https://github.com/THUSI-Lab/PrismWAM).
Its fixed source revision and file hashes are in
`third_party/prismwam/OFFICIAL_SOURCE.json`. Required notices are retained in
`third_party/prismwam/OFFICIAL_LICENSE`, `UPSTREAM_NOTICES.md` and `licenses/`.

The RoboDoctor observation adapter follows the public PrismWAM RoboDojo camera,
normalization and history contracts. The policy, RGB/depth VAEs, configuration
and action statistics come from the public
[PrismWAM RoboDojo release](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo).
Pinned download revisions and checksums are in `robodoctor/prismwam/assets.json`.
The matching text encoder and tokenizer come from the public Wan model release.

Model weights, simulators and robot assets are downloaded separately and retain
their upstream terms. This file preserves third-party notices; it does not assign
a project-wide license to RoboDoctor.
