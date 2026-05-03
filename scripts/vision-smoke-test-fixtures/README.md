# vision-smoke-test fixtures

Drop test artifacts here for `scripts/vision-smoke-test.sh`:

- `test.jpg` — a ~1MP image. A real backyard / household snapshot is ideal so the quality-sanity check is meaningful, but any image works for the timing/VRAM gates.
- `clip.mp4` — optional 5-second clip. Only used when `VIDEO=...` is exported pointing here. Skip it on first run if you don't have one handy; the smoke test will note it as skipped rather than fail.

These files are git-ignored on purpose — they are local test artifacts, not committed assets.
