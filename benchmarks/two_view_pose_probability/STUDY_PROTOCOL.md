# Academic study protocol: probabilistic two-view spherical R,t estimation

Status: preregistration draft for VAL-018. No model result may be used to alter
the splits, targets or primary metrics defined here.

## 1. Objective

Build and validate two complementary probability models for PanorAi's strictly
two-view spherical relative-pose pipeline:

1. a **capture model**, using only variables that are controllable or visible
   to the capture operator before relative-pose estimation;
2. a **post-processing model**, using diagnostics produced by detection,
   matching and R,t estimation, but never ground truth.

For capture conditions `x_capture` and post-processing evidence `x_post`, the
study estimates:

```text
p_accept_capture = P(accepted | x_capture)
p_precise_capture = P(precise | accepted, x_capture)
p_usable_capture = p_accept_capture * p_precise_capture

p_precise_post = P(precise | returned, x_capture, x_post)
```

Every estimate uses exactly two images. The number of images is therefore not
a predictor. Dataset sample size is reported as unique images, unique unordered
pairs, connected image-pair components and independent spatial/acquisition
groups.

## 2. Scientific claims and hypotheses

### Primary claim A: capture predictability

Controllable and operator-visible capture conditions predict whether the
PanorAi two-view estimator will accept a pair and whether an accepted pose will
be precise better than a dataset/group base-rate model.

### Primary claim B: calibrated post-processing confidence

Observable frontend and estimator evidence predicts precise two-view pose
better than the existing uncalibrated scalar quality score, with calibration
measured on groups and domains not used for fitting.

### Secondary claim C: domain transfer

A domain-agnostic model trained without the target dataset retains useful
calibration on Matterport360, Stanford2D3D and P74. Dataset identity may be
used to quantify heterogeneity, but it is not an operational input to the
deployed model.

### Secondary claim D: mechanistic consistency

The real-data probability surfaces are directionally consistent with the
synthetic mechanisms identified by VAL-017: increasing observable parallax and
broad angular evidence should not reduce precision after conditioning on the
remaining preregistered variables. This is a consistency check, not a license
to force monotonicity when real data contradict it.

## 3. Outcomes

### 3.1 Estimator states

- `returned`: the public estimator returns a finite R and unit translation
  direction.
- `accepted`: the unchanged public quality policy accepts the returned pose.
- `strict`: rotation error <= 5 degrees and oriented translation-direction
  error <= 10 degrees.
- `precise`: rotation error <= 1 degree and oriented translation-direction
  error <= 5 degrees.
- `catastrophic`: an accepted pose violates the preregistered catastrophic
  threshold already used by the frozen dataset evaluator.

Ground-truth errors define outcomes only. They are never model inputs.

### 3.2 Primary probabilities

- Capture acceptance model: `P(accepted | x_capture)` over every eligible pair.
- Capture selective-precision model:
  `P(precise | accepted, x_capture)` over accepted pairs only.
- Post-processing precision model:
  `P(precise | returned, x_capture, x_post)` over returned poses.

The product `p_usable_capture` is evaluated directly against
`accepted AND precise`. It is not assumed calibrated merely because its two
factors are calibrated separately.

## 4. Permitted predictors

### 4.1 Capture model: operator-visible only

Predictors must be available from capture planning, sensor metadata or a
capture-quality preview before relative-pose estimation:

- registered-cloud overlap or an explicitly identified online overlap proxy;
- physical baseline between the two capture stations;
- representative scene distance or a depth-distribution summary when the
  capture system provides depth;
- baseline/depth ratio and predicted parallax derived from the preceding
  visible quantities;
- panorama resolution and calibrated sensor profile;
- explicit valid-support fraction from masks;
- sharpness/blur, exposure clipping and photometric-discontinuity measures;
- spherical texture/keypoint count and two-dimensional angular occupancy
  computed independently for each image, before matching;
- visible dynamic/occluded-area fraction when such a capture-time detector is
  available.

Dataset identity, reference poses, true overlap unavailable to the operator,
true match correctness and all estimator outputs are forbidden operational
predictors. Dataset identity is retained only for stratified evaluation and
heterogeneity analysis.

### 4.2 Post-processing model: algorithm evidence

The second model may add quantities available after the pair has been
processed:

- match count and reciprocal/ambiguity diagnostics;
- descriptor-distance summaries;
- estimated inlier count and ratio;
- occupied equal-area cells and coverage entropy in both images;
- median and P90 epipolar residual;
- estimated median parallax;
- cheirality ratio and translation-orientation margin;
- stability trial success and R/t dispersion;
- Essential score margin and preferred competing model;
- robust trial count, refit count and degeneracy reasons.

Detector, descriptor, matching and pose runtimes are engineering diagnostics,
not probability predictors. They depend on hardware, process lifecycle and
host contention and are analyzed only in the separately controlled timing
protocol.

Ground-truth geometric-match fraction, reference parallax, reference overlap
not visible at deployment, and R/t errors are forbidden.

## 5. Dataset census and independence

The first experiment is an audit, not model fitting.

For each dataset, record:

- unique source images;
- unique unordered pairs after removing reversed duplicates;
- number of images reused by more than one pair and maximum pair degree;
- connected components of the image-pair graph;
- independent buildings, areas or acquisition families;
- returned, accepted, strict and precise counts;
- missingness for every predictor.

Known starting evidence is:

| Dataset | Frozen pairs | Known independent groups | Current role |
| --- | ---: | ---: | --- |
| Matterport360 | 1,890 | 63 | primary inference |
| Stanford2D3D | 450 | 3 | low-powered external replication |
| P74 overlap-response sample | 45 | 3 | industrial development evidence |

The current P74 sample contains 74 unique images. Matterport and Stanford
unique-image counts must be recomputed from the canonical pair manifests; they
must not be inferred as twice the pair count.

Any two pairs sharing an image must remain in the same split. When image-pair
components cross nominal spatial groups, the connected component becomes the
split unit. Bootstrap resampling uses independent groups/components, never
individual pairs.

## 6. Experiments

### E0 — corpus census and leakage audit

Inputs: canonical pair manifests and frozen evaluator outcomes.

Outputs:

- one standardized row per unordered pair;
- dataset census and missingness report;
- duplicate/reversal audit;
- image-pair graph components;
- frozen group-safe development, calibration and evaluation assignments;
- SHA-256 hashes for every input and derived table.

Gate: no duplicated unordered pair, no shared image across splits, no outcome
field present in the prediction-only table.

### E1 — capture-variable measurement validation

Verify each `x_capture` variable against its operational source. For every
variable, record whether it is directly controlled, directly visible, derived
from visible inputs or unavailable in a dataset. Compare online proxies with
reference-only values without allowing the reference value into the model.

Gate: the capture model uses only variables marked operationally available in
all datasets included in that fitted variant. Missing depth produces a named
RGB-only model rather than silent imputation from ground truth.

### E2 — capture acceptance probability

Fit regularized logistic models for `accepted` using nested group-aware
validation. Compare:

1. group/dataset base rate;
2. baseline and overlap only;
3. geometry plus image-quality variables;
4. geometry plus image-quality plus per-image spherical texture distribution.

Continuous terms use transformations frozen from development data. Nonlinear
terms are limited to preregistered bounded splines and physically motivated
interactions such as baseline/depth with overlap.

Primary metrics: group-held-out Brier score and log loss. Secondary metrics:
ECE, calibration intercept/slope, discrimination and acceptance prevalence.

### E3 — capture conditional precision probability

On accepted pairs only, fit `P(precise | accepted, x_capture)` using the same
splits and predictor blocks as E2. Report both selective precision and coverage
so that abstention cannot inflate the apparent result silently.

Evaluate `p_usable_capture` against `accepted AND precise` on the untouched
evaluation groups.

### E4 — post-processing precision probability

On every returned pose, add `x_post` in ordered blocks:

1. current raw quality score baseline;
2. correspondence support and angular coverage;
3. residual, parallax and cheirality evidence;
4. stability and model-competition evidence;
5. complete preregistered post-processing model.

This ablation identifies which evidence materially improves calibration. A
more complex block is retained only when it improves group-held-out Brier and
log loss without reducing precision at the chosen deployment coverage.

### E5 — cross-dataset transfer

Run all three leave-one-dataset-out directions. The deployed transfer model
cannot receive the held-out dataset label. Report each domain separately and a
macro average giving equal weight to the three datasets. Pair-weighted pooled
metrics are secondary only.

Because Stanford and P74 currently contain only three independent groups each,
their intervals and calibration plots are descriptive. They cannot close a
cross-domain release claim without additional independent groups.

### E6 — overlap and capture-envelope response

For the capture model, render probability surfaces against spatial overlap and
baseline/depth or predicted parallax, holding remaining operator-visible
variables at documented reference values. Report uncertainty bands and the
number of independent groups supporting every region.

No minimum overlap threshold is chosen from a cell containing fewer than five
independent groups. Sparse regions are marked unsupported, not interpolated as
reliable.

### E7 — selective release rule

Freeze probability thresholds on development/calibration groups, then evaluate
once on untouched groups. A release rule must state simultaneously:

- probability threshold;
- accepted coverage;
- precise-pose precision;
- catastrophic accepted count;
- one-sided lower confidence bound;
- supported domains and capture-variable ranges.

The final recommendation is the intersection of the capture envelope and the
post-processing confidence gate. Neither model may silently compensate for an
unsupported condition in the other.

### E8 — prospective confirmation

After all thresholds are frozen, collect or designate new independent capture
groups. Run prediction before opening reference poses. This phase is the only
one eligible to support a confirmatory academic/release claim.

## 7. Model and calibration protocol

The primary estimator is a regularized logistic model with transparent frozen
feature transforms. A bounded spline/GAM variant may be compared, but opaque
high-capacity models are secondary unless the data volume and group count
support them.

Calibration candidates are Platt/logistic recalibration and isotonic
regression. Calibration is fit only inside each outer training fold. The
evaluation fold is never used to choose model family, features, regularization,
thresholds or calibration method.

All predictions are saved before evaluation as immutable pair-ID/probability
tables. Evaluation joins ground truth only after verifying order, uniqueness,
hashes and split membership.

## 8. Metrics and uncertainty

Primary probability metrics:

- Brier score;
- logarithmic loss;
- calibration intercept and slope;
- ECE with fixed, preregistered bins;
- reliability curves with counts and group support per bin.

Operational metrics:

- return and acceptance coverage;
- strict and precise accuracy over all eligible pairs;
- precision among accepted/selected pairs;
- catastrophic false-accept rate;
- `p_usable` calibration.

Confidence intervals use spatial/acquisition-group bootstrap. When a dataset
has too few groups for meaningful bootstrap inference, report exact counts and
Wilson/binomial intervals as descriptive diagnostics without claiming group-
generalized confidence.

## 9. Statistical power and minimum evidence

Pair count is not the effective sample size when images or environments are
reused. Power is assessed from the number of independent groups, group-size
distribution, outcome prevalence and intragroup correlation measured during
E0.

As an optimistic independent-pair reference only, observing zero failures in
59 accepted precise pairs is required for a one-sided 95% exact lower bound of
approximately 95% precision. Clustering raises the required pair count. The
actual prospective sample target will be computed by simulation from E0's
group structure and frozen before E8.

Matterport's 63 groups can support primary group-aware inference. Stanford and
P74, with three groups each in current evidence, support replication and
failure discovery but not a strong standalone probability claim. Expanding
the number of independent Stanford areas is constrained by the dataset;
expanding P74 requires new acquisition families or separately held sequences,
not more correlated pairs from the same three families.

## 10. Academic deliverables

The study must produce:

1. a dataset census table at image, pair and group levels;
2. a complete operational-variable data dictionary;
3. a reproducible frozen split manifest;
4. capture and post-processing model cards;
5. calibration and probability-response figures with support counts;
6. feature-block ablation tables;
7. leave-one-dataset-out transfer results;
8. a failure taxonomy with representative real pairs;
9. a prospective confirmation report;
10. a paper-ready methods/results document that distinguishes exploratory,
   validation and confirmatory evidence.

No academic or release conclusion may pool synthetic and real samples as if
they were exchangeable, count correlated pairs as independent, or describe a
post-hoc P74 threshold as held-out confirmation.
