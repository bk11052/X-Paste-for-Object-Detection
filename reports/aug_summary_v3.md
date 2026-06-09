# Aug provenance summary

## Reference: train GT class distribution

| class | GT count | share |
|---|---:|---:|
| Soldier | 577 | 26.0% |
| civilian_vehicle | 424 | 19.1% |
| military_vehicle | 363 | 16.4% |
| persons | 856 | 38.6% |

_Total GT instances: 2220; train images: 1352_

## Summary across modes

| mode | imgs | accept rate | total pastes | pastes / accepted img | unique pool inst |
|---|---:|---:|---:|---:|---:|
| B | 1352 | 100.0% | 1755 | 1.30 | 876 |
| C | 1352 | 100.0% | 1755 | 1.30 | 83 |
| D | 1352 | 53.1% | 973 | 1.36 | 65 |
| E | 1352 | 53.1% | 910 | 1.27 | 57 |

## Per-class paste counts (and share of total pastes within mode)

| mode | Soldier | civilian_vehicle | military_vehicle | persons | total |
|---|---:|---:|---:|---:|---:|
| B | 180 (10.3%) | 507 (28.9%) | 461 (26.3%) | 607 (34.6%) | 1755 |
| C | 180 (10.3%) | 507 (28.9%) | 461 (26.3%) | 607 (34.6%) | 1755 |
| D | 190 (19.5%) | 230 (23.6%) | 201 (20.7%) | 352 (36.2%) | 973 |
| E | 262 (28.8%) | 197 (21.6%) | 205 (22.5%) | 246 (27.0%) | 910 |

**Inverse-freq evidence**: in mode E, the share of underrepresented classes (per the train GT histogram above) should be higher than in mode D (uniform). If E's share for the rarest class exceeds D's by ≥5 pp, the inverse-freq sampler is doing what we claim.

## Scale bin distribution (q0–q25 | q25–q50 | q50–q75 | q75+)

| mode | bin 0 | bin 1 | bin 2 | bin 3 |
|---|---:|---:|---:|---:|
| B | 1427 (81.3%) | 280 (16.0%) | 48 (2.7%) | 0 (0.0%) |
| C | 1427 (81.3%) | 280 (16.0%) | 48 (2.7%) | 0 (0.0%) |
| D | 803 (82.5%) | 149 (15.3%) | 21 (2.2%) | 0 (0.0%) |
| E | 223 (24.5%) | 258 (28.4%) | 228 (25.1%) | 201 (22.1%) |

**Scale tail oversampling**: bins 0 and 3 are the tails. If E's (bin 0 + bin 3) share exceeds D's, the sampler is biasing toward small + large objects (paper claim).

## cx bin distribution (left → right)

| mode | bin 0 | bin 1 | bin 2 | bin 3 |
|---|---:|---:|---:|---:|
| B | 359 (20.5%) | 516 (29.4%) | 503 (28.7%) | 377 (21.5%) |
| C | 359 (20.5%) | 516 (29.4%) | 503 (28.7%) | 377 (21.5%) |
| D | 194 (19.9%) | 294 (30.2%) | 288 (29.6%) | 197 (20.2%) |
| E | 238 (26.2%) | 194 (21.3%) | 206 (22.6%) | 272 (29.9%) |

## cy bin distribution (top → bottom)

| mode | bin 0 | bin 1 | bin 2 | bin 3 |
|---|---:|---:|---:|---:|
| B | 0 (0.0%) | 412 (23.5%) | 835 (47.6%) | 508 (28.9%) |
| C | 0 (0.0%) | 412 (23.5%) | 835 (47.6%) | 508 (28.9%) |
| D | 0 (0.0%) | 237 (24.4%) | 455 (46.8%) | 281 (28.9%) |
| E | 453 (49.8%) | 169 (18.6%) | 72 (7.9%) | 216 (23.7%) |

## Scale method distribution (depth-free fallback chain)

| mode | cy_regression | gt_anchor | target_scale | total |
|---|---:|---:|---:|---:|
| B | 32 (1.8%) | 445 (25.4%) | 1278 (72.8%) | 1755 |
| C | 32 (1.8%) | 445 (25.4%) | 1278 (72.8%) | 1755 |
| D | 95 (9.8%) | 281 (28.9%) | 597 (61.4%) | 973 |
| E | 121 (13.3%) | 528 (58.0%) | 261 (28.7%) | 910 |

`gt_anchor` = scaled from a same-class GT box on this image; `cy_regression` = predicted from per-class cy → h fit; `target_scale` = fallback to sampled scale bin midpoint.

## Host accept/reject

| mode | accepted | gt_area_too_dense (0.500) | gt_area_too_dense (0.503) | gt_area_too_dense (0.504) | gt_area_too_dense (0.505) | gt_area_too_dense (0.506) | gt_area_too_dense (0.507) | gt_area_too_dense (0.509) | gt_area_too_dense (0.510) | gt_area_too_dense (0.514) | gt_area_too_dense (0.515) | gt_area_too_dense (0.516) | gt_area_too_dense (0.517) | gt_area_too_dense (0.520) | gt_area_too_dense (0.521) | gt_area_too_dense (0.523) | gt_area_too_dense (0.525) | gt_area_too_dense (0.526) | gt_area_too_dense (0.528) | gt_area_too_dense (0.529) | gt_area_too_dense (0.530) | gt_area_too_dense (0.531) | gt_area_too_dense (0.532) | gt_area_too_dense (0.533) | gt_area_too_dense (0.534) | gt_area_too_dense (0.535) | gt_area_too_dense (0.536) | gt_area_too_dense (0.538) | gt_area_too_dense (0.539) | gt_area_too_dense (0.541) | gt_area_too_dense (0.543) | gt_area_too_dense (0.544) | gt_area_too_dense (0.548) | gt_area_too_dense (0.550) | gt_area_too_dense (0.551) | gt_area_too_dense (0.552) | gt_area_too_dense (0.555) | gt_area_too_dense (0.556) | gt_area_too_dense (0.558) | gt_area_too_dense (0.559) | gt_area_too_dense (0.560) | gt_area_too_dense (0.561) | gt_area_too_dense (0.562) | gt_area_too_dense (0.563) | gt_area_too_dense (0.566) | gt_area_too_dense (0.569) | gt_area_too_dense (0.570) | gt_area_too_dense (0.571) | gt_area_too_dense (0.573) | gt_area_too_dense (0.574) | gt_area_too_dense (0.575) | gt_area_too_dense (0.577) | gt_area_too_dense (0.579) | gt_area_too_dense (0.580) | gt_area_too_dense (0.581) | gt_area_too_dense (0.582) | gt_area_too_dense (0.583) | gt_area_too_dense (0.585) | gt_area_too_dense (0.586) | gt_area_too_dense (0.587) | gt_area_too_dense (0.589) | gt_area_too_dense (0.590) | gt_area_too_dense (0.591) | gt_area_too_dense (0.592) | gt_area_too_dense (0.593) | gt_area_too_dense (0.594) | gt_area_too_dense (0.596) | gt_area_too_dense (0.597) | gt_area_too_dense (0.600) | gt_area_too_dense (0.602) | gt_area_too_dense (0.603) | gt_area_too_dense (0.605) | gt_area_too_dense (0.606) | gt_area_too_dense (0.608) | gt_area_too_dense (0.609) | gt_area_too_dense (0.610) | gt_area_too_dense (0.611) | gt_area_too_dense (0.612) | gt_area_too_dense (0.615) | gt_area_too_dense (0.616) | gt_area_too_dense (0.617) | gt_area_too_dense (0.618) | gt_area_too_dense (0.619) | gt_area_too_dense (0.620) | gt_area_too_dense (0.621) | gt_area_too_dense (0.623) | gt_area_too_dense (0.626) | gt_area_too_dense (0.627) | gt_area_too_dense (0.628) | gt_area_too_dense (0.629) | gt_area_too_dense (0.635) | gt_area_too_dense (0.637) | gt_area_too_dense (0.639) | gt_area_too_dense (0.640) | gt_area_too_dense (0.641) | gt_area_too_dense (0.646) | gt_area_too_dense (0.650) | gt_area_too_dense (0.651) | gt_area_too_dense (0.652) | gt_area_too_dense (0.654) | gt_area_too_dense (0.655) | gt_area_too_dense (0.656) | gt_area_too_dense (0.658) | gt_area_too_dense (0.661) | gt_area_too_dense (0.663) | gt_area_too_dense (0.664) | gt_area_too_dense (0.665) | gt_area_too_dense (0.668) | gt_area_too_dense (0.671) | gt_area_too_dense (0.672) | gt_area_too_dense (0.673) | gt_area_too_dense (0.677) | gt_area_too_dense (0.678) | gt_area_too_dense (0.680) | gt_area_too_dense (0.681) | gt_area_too_dense (0.682) | gt_area_too_dense (0.683) | gt_area_too_dense (0.685) | gt_area_too_dense (0.687) | gt_area_too_dense (0.688) | gt_area_too_dense (0.689) | gt_area_too_dense (0.691) | gt_area_too_dense (0.693) | gt_area_too_dense (0.694) | gt_area_too_dense (0.695) | gt_area_too_dense (0.696) | gt_area_too_dense (0.699) | gt_area_too_dense (0.701) | gt_area_too_dense (0.702) | gt_area_too_dense (0.703) | gt_area_too_dense (0.704) | gt_area_too_dense (0.705) | gt_area_too_dense (0.706) | gt_area_too_dense (0.709) | gt_area_too_dense (0.710) | gt_area_too_dense (0.711) | gt_area_too_dense (0.712) | gt_area_too_dense (0.713) | gt_area_too_dense (0.714) | gt_area_too_dense (0.715) | gt_area_too_dense (0.716) | gt_area_too_dense (0.717) | gt_area_too_dense (0.720) | gt_area_too_dense (0.721) | gt_area_too_dense (0.723) | gt_area_too_dense (0.726) | gt_area_too_dense (0.731) | gt_area_too_dense (0.733) | gt_area_too_dense (0.741) | gt_area_too_dense (0.743) | gt_area_too_dense (0.745) | gt_area_too_dense (0.747) | gt_area_too_dense (0.748) | gt_area_too_dense (0.749) | gt_area_too_dense (0.751) | gt_area_too_dense (0.753) | gt_area_too_dense (0.754) | gt_area_too_dense (0.757) | gt_area_too_dense (0.758) | gt_area_too_dense (0.759) | gt_area_too_dense (0.762) | gt_area_too_dense (0.763) | gt_area_too_dense (0.770) | gt_area_too_dense (0.771) | gt_area_too_dense (0.774) | gt_area_too_dense (0.775) | gt_area_too_dense (0.777) | gt_area_too_dense (0.778) | gt_area_too_dense (0.780) | gt_area_too_dense (0.781) | gt_area_too_dense (0.782) | gt_area_too_dense (0.784) | gt_area_too_dense (0.785) | gt_area_too_dense (0.788) | gt_area_too_dense (0.790) | gt_area_too_dense (0.794) | gt_area_too_dense (0.797) | gt_area_too_dense (0.798) | gt_area_too_dense (0.801) | gt_area_too_dense (0.803) | gt_area_too_dense (0.806) | gt_area_too_dense (0.808) | gt_area_too_dense (0.814) | gt_area_too_dense (0.817) | gt_area_too_dense (0.819) | gt_area_too_dense (0.828) | gt_area_too_dense (0.829) | gt_area_too_dense (0.833) | gt_area_too_dense (0.835) | gt_area_too_dense (0.836) | gt_area_too_dense (0.838) | gt_area_too_dense (0.839) | gt_area_too_dense (0.840) | gt_area_too_dense (0.843) | gt_area_too_dense (0.851) | gt_area_too_dense (0.852) | gt_area_too_dense (0.854) | gt_area_too_dense (0.856) | gt_area_too_dense (0.861) | gt_area_too_dense (0.864) | gt_area_too_dense (0.865) | gt_area_too_dense (0.868) | gt_area_too_dense (0.871) | gt_area_too_dense (0.874) | gt_area_too_dense (0.875) | gt_area_too_dense (0.877) | gt_area_too_dense (0.882) | gt_area_too_dense (0.884) | gt_area_too_dense (0.902) | gt_area_too_dense (0.910) | gt_area_too_dense (0.911) | gt_area_too_dense (0.913) | gt_area_too_dense (0.916) | gt_area_too_dense (0.931) | gt_area_too_dense (0.938) | gt_area_too_dense (0.950) | gt_area_too_dense (0.956) | gt_area_too_dense (0.962) | gt_area_too_dense (0.973) | gt_area_too_dense (0.974) | gt_area_too_dense (1.011) | gt_area_too_dense (1.023) | gt_area_too_dense (1.100) | gt_area_too_dense (1.148) | gt_area_too_dense (1.223) | gt_area_too_dense (1.265) | gt_count_too_high (10) | gt_count_too_high (11) | gt_count_too_high (14) | gt_count_too_high (9) | insufficient_paste_region (ground+road=0.000) | insufficient_paste_region (ground+road=0.001) | insufficient_paste_region (ground+road=0.002) | insufficient_paste_region (ground+road=0.003) | insufficient_paste_region (ground+road=0.004) | insufficient_paste_region (ground+road=0.005) | insufficient_paste_region (ground+road=0.006) | insufficient_paste_region (ground+road=0.007) | insufficient_paste_region (ground+road=0.008) | insufficient_paste_region (ground+road=0.009) | insufficient_paste_region (ground+road=0.010) | insufficient_paste_region (ground+road=0.011) | insufficient_paste_region (ground+road=0.012) | insufficient_paste_region (ground+road=0.013) | insufficient_paste_region (ground+road=0.014) | insufficient_paste_region (ground+road=0.016) | insufficient_paste_region (ground+road=0.017) | insufficient_paste_region (ground+road=0.020) | insufficient_paste_region (ground+road=0.021) | insufficient_paste_region (ground+road=0.025) | insufficient_paste_region (ground+road=0.026) | insufficient_paste_region (ground+road=0.027) | insufficient_paste_region (ground+road=0.028) | insufficient_paste_region (ground+road=0.029) | insufficient_paste_region (ground+road=0.030) | insufficient_paste_region (ground+road=0.031) | insufficient_paste_region (ground+road=0.032) | insufficient_paste_region (ground+road=0.033) | insufficient_paste_region (ground+road=0.034) | insufficient_paste_region (ground+road=0.039) | insufficient_paste_region (ground+road=0.040) | insufficient_paste_region (ground+road=0.043) | insufficient_paste_region (ground+road=0.044) | insufficient_paste_region (ground+road=0.045) | insufficient_paste_region (ground+road=0.046) | insufficient_paste_region (ground+road=0.047) | insufficient_paste_region (ground+road=0.048) | insufficient_paste_region (ground+road=0.052) | insufficient_paste_region (ground+road=0.053) | insufficient_paste_region (ground+road=0.054) | insufficient_paste_region (ground+road=0.055) | insufficient_paste_region (ground+road=0.056) | insufficient_paste_region (ground+road=0.058) | insufficient_paste_region (ground+road=0.059) | insufficient_paste_region (ground+road=0.061) | insufficient_paste_region (ground+road=0.062) | insufficient_paste_region (ground+road=0.063) | insufficient_paste_region (ground+road=0.064) | insufficient_paste_region (ground+road=0.065) | insufficient_paste_region (ground+road=0.067) | insufficient_paste_region (ground+road=0.068) | insufficient_paste_region (ground+road=0.069) | insufficient_paste_region (ground+road=0.070) | insufficient_paste_region (ground+road=0.071) | insufficient_paste_region (ground+road=0.072) | insufficient_paste_region (ground+road=0.074) | insufficient_paste_region (ground+road=0.075) | insufficient_paste_region (ground+road=0.076) | insufficient_paste_region (ground+road=0.078) | insufficient_paste_region (ground+road=0.079) | insufficient_paste_region (ground+road=0.080) | insufficient_paste_region (ground+road=0.081) | insufficient_paste_region (ground+road=0.082) | insufficient_paste_region (ground+road=0.085) | insufficient_paste_region (ground+road=0.086) | insufficient_paste_region (ground+road=0.087) | insufficient_paste_region (ground+road=0.088) | insufficient_paste_region (ground+road=0.089) | insufficient_paste_region (ground+road=0.092) | insufficient_paste_region (ground+road=0.093) | insufficient_paste_region (ground+road=0.094) | insufficient_paste_region (ground+road=0.095) | insufficient_paste_region (ground+road=0.096) | insufficient_paste_region (ground+road=0.097) | insufficient_paste_region (ground+road=0.098) | insufficient_paste_region (ground+road=0.099) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 1352 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C | 1352 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D | 718 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 4 | 1 | 3 | 2 | 3 | 2 | 3 | 1 | 2 | 4 | 1 | 1 | 1 | 2 | 3 | 1 | 1 | 3 | 1 | 3 | 1 | 1 | 1 | 4 | 2 | 3 | 2 | 1 | 2 | 1 | 1 | 1 | 3 | 1 | 3 | 2 | 1 | 4 | 1 | 1 | 3 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 3 | 1 | 1 | 1 | 3 | 2 | 1 | 3 | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 1 | 2 | 2 | 1 | 2 | 2 | 1 | 2 | 1 | 1 | 1 | 3 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 2 | 1 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 4 | 2 | 1 | 1 | 109 | 9 | 8 | 5 | 6 | 6 | 5 | 4 | 3 | 7 | 4 | 4 | 3 | 1 | 3 | 1 | 2 | 1 | 4 | 1 | 2 | 3 | 2 | 3 | 2 | 5 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 3 | 2 | 5 | 3 | 3 | 1 | 3 | 2 | 3 | 1 | 1 | 3 | 2 | 2 | 1 | 2 | 4 | 3 | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 2 | 2 | 6 | 1 | 1 | 1 | 4 | 3 | 4 | 1 | 2 |
| E | 718 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 4 | 1 | 3 | 2 | 3 | 2 | 3 | 1 | 2 | 4 | 1 | 1 | 1 | 2 | 3 | 1 | 1 | 3 | 1 | 3 | 1 | 1 | 1 | 4 | 2 | 3 | 2 | 1 | 2 | 1 | 1 | 1 | 3 | 1 | 3 | 2 | 1 | 4 | 1 | 1 | 3 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 3 | 1 | 1 | 1 | 3 | 2 | 1 | 3 | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 1 | 2 | 2 | 1 | 2 | 2 | 1 | 2 | 1 | 1 | 1 | 3 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 2 | 1 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 4 | 2 | 1 | 1 | 109 | 9 | 8 | 5 | 6 | 6 | 5 | 4 | 3 | 7 | 4 | 4 | 3 | 1 | 3 | 1 | 2 | 1 | 4 | 1 | 2 | 3 | 2 | 3 | 2 | 5 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 3 | 2 | 5 | 3 | 3 | 1 | 3 | 2 | 3 | 1 | 1 | 3 | 2 | 2 | 1 | 2 | 4 | 3 | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 2 | 2 | 6 | 1 | 1 | 1 | 4 | 3 | 4 | 1 | 2 |

## Top-10 reused SD pool instances per mode

### B
| rank | filename | reuse count |
|---:|---|---:|
| 1 | `Person000000006293_jpg.rf.f06e1d1de545ade4aa4337667ba2a4a2.jpg` | 16 |
| 2 | `Person000000006811_jpg.rf.a2e98c08205af1bbb69a833c7174af2d.jpg` | 12 |
| 3 | `Austin-Maxi-1750-02-06-13-026_jpg.rf.9d4bb54fd93e4b738bea6528d161c64b.jpg` | 12 |
| 4 | `Person000000003077_jpg.rf.618ac394968b2dc898f999c3ca190f75.jpg` | 11 |
| 5 | `Person000000005388_jpg.rf.cedf2bb4ea20f60ad42a2d5dfacd649c.jpg` | 9 |
| 6 | `frame-id-36444fca-9bb7-11ed-9981-59abf0a4b5f1_jpg.rf.3e54adc89bbe6102e0ead51abc074d46.jpg` | 9 |
| 7 | `Er7NgTIWMAE4kZW_jpeg_jpg.rf.a951505e43067258c3fa4901a364680e.jpg` | 9 |
| 8 | `Person000000005260_jpg.rf.7dc56430d48422eb642bcc19e17e9b5e.jpg` | 8 |
| 9 | `Person000000004359_jpg.rf.1cc4ca6da39551864324409c6c738697.jpg` | 8 |
| 10 | `Person000000003124_jpg.rf.fb6e20d35a3aa42638c77b2726dbb1bc.jpg` | 8 |

### C
| rank | filename | reuse count |
|---:|---|---:|
| 1 | `0003.png` | 75 |
| 2 | `0009.png` | 71 |
| 3 | `0005.png` | 70 |
| 4 | `0000.png` | 66 |
| 5 | `0017.png` | 66 |
| 6 | `0006.png` | 66 |
| 7 | `0008.png` | 65 |
| 8 | `0002.png` | 64 |
| 9 | `0004.png` | 64 |
| 10 | `0001.png` | 63 |

### D
| rank | filename | reuse count |
|---:|---|---:|
| 1 | `0009.png` | 62 |
| 2 | `0003.png` | 52 |
| 3 | `0006.png` | 47 |
| 4 | `0000.png` | 44 |
| 5 | `0008.png` | 41 |
| 6 | `0011.png` | 40 |
| 7 | `0004.png` | 40 |
| 8 | `0013.png` | 38 |
| 9 | `0015.png` | 35 |
| 10 | `0002.png` | 33 |

### E
| rank | filename | reuse count |
|---:|---|---:|
| 1 | `0005.png` | 53 |
| 2 | `0009.png` | 52 |
| 3 | `0022.png` | 43 |
| 4 | `0008.png` | 42 |
| 5 | `0013.png` | 39 |
| 6 | `0000.png` | 39 |
| 7 | `0006.png` | 38 |
| 8 | `0017.png` | 36 |
| 9 | `0016.png` | 34 |
| 10 | `0018.png` | 33 |
