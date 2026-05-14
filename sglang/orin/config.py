"""
SGLang package configuration — Jetson Orin AGX variant (sm_87, JetPack 6.x, CUDA 12.6)

Vendored from dusty-nv/jetson-containers:packages/llm/sglang/config.py
(see ../LICENSE-UPSTREAM.md for attribution), with the version pin
diverged from upstream's CUDA-13-tied 0.5.11 to a JP6/CUDA-12.6-
compatible 0.5.x release that ships gpt_oss.py in srt/models/.

Pin rationale (INFR-74 + INFR-77):
  - Upstream's current default (0.5.11) is annotated "Compatible with
    CUDA 13 (Spark and Thor)" — won't build for JetPack 6 / CUDA 12.6.
  - Dustynv's last Orin-targeted published tag (r36.4.0) shipped
    0.4.1.post7 (Feb 2025), which predates gpt-oss support
    (Aug 2025) and has no srt/models/gpt_oss.py.
  - 0.5.3 is the initial pick: first 0.5.x line with gpt-oss model
    class, predates the upstream's CUDA-13 transition (which landed
    around 0.5.11). On-target smoke build on Hephaestus is the gate.
  - Fallback ladder if 0.5.3 doesn't build: 0.5.2 → 0.5.1 → 0.5.0,
    then 0.4.5+. Document the working pin back here when verified.
"""
from jetson_containers import CUDA_VERSION, IS_SBSA, update_dependencies
from packaging.version import Version


def sglang(version, version_spec=None, requires=None, depends=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    pkg['name'] = f'sglang:{version}'

    pkg['build_args'] = {
        'SGLANG_VERSION': version,
        'SGLANG_VERSION_SPEC': version_spec,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'sglang:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sglang'
        builder['alias'] = 'sglang:builder'

    return pkg, builder


package = [
    sglang(
        '0.5.3',
        '0.5.3',
        depends=['flashinfer', 'sgl-kernel:0.5.3', 'torchao:0.17.0'],
        default=True,
    ),  # Orin/JP6/CUDA 12.6 pin — first 0.5.x with gpt_oss.py; verify via on-target smoke build (INFR-77)
]
