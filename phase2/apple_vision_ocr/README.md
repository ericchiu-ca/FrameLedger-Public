# Apple Vision OCR helper

This Phase 2 helper is intentionally separate from the frozen Phase 1 frame
selection code. Its primary interface reads one JSON request for exactly one PNG
from standard input, performs local Apple Vision text recognition, and writes one
strict JSON response to standard output. Errors are JSON on standard error and
leave standard output empty. It has no third-party dependencies and does not
contact a network service.

## Build

```bash
xcrun swiftc \
  phase2/apple_vision_ocr/main.swift \
  -o /tmp/frameledger-apple-vision-ocr \
  -framework Vision \
  -framework CoreGraphics \
  -framework CoreImage \
  -framework ImageIO
```

The helper requires macOS 12 or later because it checks the requested BCP-47
language tags against the languages supported by the configured Vision request.

## Run

```bash
printf '%s' '{
  "protocol":"frameledger-ocr-helper-v1",
  "image_path":"/absolute/path/frame.png",
  "languages":["zh-Hans","en-US"],
  "route_kind":"presentation",
  "roi_normalized":[0.04,0.02,0.94,0.88],
  "bbox_origin":"top_left_normalized"
}' | /tmp/frameledger-apple-vision-ocr
```

The executable must receive no arguments when called by FrameLedger. The legacy
manual form `--image IMAGE.png --languages zh-Hans,en-US` is retained as a
full-image diagnostic. Duplicate tags are folded while retaining their first
position because Vision uses language order as recognition priority.

On success, stdout has this contract:

```json
{
  "engine": {
    "bbox_origin": "top_left_normalized",
    "current_revision": 3,
    "operating_system": {"major": 27, "minor": 0, "patch": 0},
    "recognition_languages": ["zh-Hans", "en-US"],
    "requested_roi_normalized": [0.04, 0.02, 0.94, 0.88],
    "request_revision": 3
  },
  "observations": [
    {
      "bbox": [0.08, 0.11, 0.12, 0.03],
      "confidence": 0.97,
      "order": 0,
      "text": "example",
      "vision_index": 0
    }
  ],
  "protocol": "frameledger-ocr-helper-v1"
}
```

Vision runs only on the requested ROI. Its ROI-local, bottom-left boxes are
converted back to whole-image top-left normalized `[x, y, width, height]`, with
the effective pixel-aligned ROI offset applied. Coordinates/confidence are
clamped and rounded to six decimal places. Results are sorted top-to-bottom then
left-to-right and assigned zero-based `order`; `vision_index` preserves the
framework's original result position.

The engine object also records the actual/default/current request revision,
supported revisions, Vision framework version/build when exposed by the OS, the
recognition settings, and the structured macOS version. These fields must travel
with downstream OCR evidence because Vision behavior can change across OS or
request revisions.

## Tests and sandbox behavior

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest \
  tests.test_apple_vision_ocr_helper -v
```

The test compiles the helper in a temporary directory, checks failure-channel and
JSON contracts, and performs one real Vision request against a generated PNG.
Some restricted execution environments let the helper compile but make Vision
fail with native code `8`, or wrap the same runtime denial as
`Foundation._GenericObjCError` code `0`. The smoke test reports only those
explicit conditions as a skip rather than claiming OCR succeeded; any other
runtime failure remains a test failure.
