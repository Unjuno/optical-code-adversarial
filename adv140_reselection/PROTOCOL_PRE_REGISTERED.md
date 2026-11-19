# 140B-only auxiliary 42-check-row reselection protocol

Recorded before any new-matrix evaluation against the 1,000 frozen final-validation samples.

## Frozen experimental specification
- Source: `/mnt/data/optical_code_adversarial_v02.zip`, extracted under `optical_code_adversarial_v02/` (Git clone unavailable because DNS resolution for github.com fails in this container).
- Reference `advgraph140`: 138-byte adversarial core plus 2-byte auxiliary information; CRC16 on 2-byte auxiliary; systematic auxiliary 74-bit code = 32 systematic bits (16 data + 16 CRC) + 42 parity bits; 37 RGB4 modules.
- Change **only** the `42 x 32` auxiliary parity incidence matrix P. Effective parity-check H = [P | I_42]. Do not change core parity (18 min-sum iterations), auxiliary min-sum (24 iterations; alpha=0.8), RGB palette, layout, detector threshold 140, channel, or size.
- Selection sees exclusively *development* streams: payload generator seeds `341000000+i`, hard-camera seeds `441000000+i` and challenge-camera seeds `541000000+i`. No seed equal to the held-out seeds below. Seeded pseudo-random candidate matrices use deterministic random states.
- Final test (not used for selection): exactly same old report holdout: 1,000 items `payload_seed=121100000+i`, `channel_seed=222100000+i`, `condition=hard`, `i=0..999`. Both arms receive the identical data and renderer seed.
- Baseline 928/1000 from the frozen report is known in advance, so this is a repeated (not never-seen-by-anyone) holdout; candidate selection may **not** inspect per-item holdout outcomes or scores before freezing the matrix.

## Predeclared decision rule
- Unit of analysis: 1 complete 140-byte message from one synthetic camera frame; primary endpoint is whole-message exact-byte recovery. Additional-CRC failure is measured only when the core was accepted by its CRC, with separate counts of core_parity, core_crc and detection failures.
- PASS: (i) new complete-file success is strictly greater than reference, (ii) aux_crc failures strictly fewer, (iii) one-sided exact paired McNemar/sign test for improvement on the common core-accepted items has p<0.05, and (iv) no change to frozen components/parameters.
- FAIL: complete-message success is <= reference, or aux_crc failure count is >= reference, or frozen component changes.
- UNCERTAIN: success rises and aux_crc falls but the one-sided paired test is not significant at alpha 0.05.
- Report complete recovery 95% Wilson score intervals for each arm, not only an unpaired z-test. Also report both discordance counts and all failure classifications.
- No early stopping on final 1,000 test cases. If interrupted, report as incomplete/UNCERTAIN, not a completed outcome.

## Limitations
- This is a synthetic RGB camera-only paired test, not real-camera or ISO QR/JAB benchmark.
- Changed auxiliary parity symbols can in principle perturb downstream image detection even though fixed frame/pilots stay identical; observe actual outcomes rather than assuming exact invariance.
- Selection is driven by developmental performance, not by previously published final-data per-case results.
