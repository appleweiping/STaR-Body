"""Build the additive stride-4 overlapping variant of the v5 T17 cache.

The implementation is intentionally reused from the tested v5 builder while
this entry point changes only the cache format and target stride.  Eight-frame
targets overlap by four frames, so every valid acceleration triplet is present
inside at least one training window.  All model files remain unchanged by this
variant.
"""

from __future__ import annotations

import cache_continuous_3dpw_vace_features_v5 as implementation


FORMAT = implementation.STRIDE4_FORMAT
TARGET_STRIDE = 4


def main() -> None:
    implementation.FORMAT = FORMAT
    implementation.TARGET_STRIDE = TARGET_STRIDE
    implementation.main()


if __name__ == "__main__":
    main()
