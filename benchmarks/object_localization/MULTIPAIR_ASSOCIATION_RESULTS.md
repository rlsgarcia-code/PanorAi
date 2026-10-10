# Multipair object-association validation

## Result

The bounded end-to-end development gate is **FAIL**. Association remained
precision-first and fail-closed, but the current feature/relative-pose front
end did not expose enough eligible evidence to meet the frozen recall and pose
coverage gates.

Across six real Stanford2D3D panorama pairs from three areas and five office or
lobby scenes, the base run contained 14 visible cross-view object identities:

| Measure | Result |
| --- | ---: |
| Correctly promoted identities | 8 / 14 |
| False promotions | 0 |
| Base precision | 100% |
| Base recall | 57.1% |
| Pairs with accepted relative pose | 4 / 6 |
| Deterministic ID repetitions | 24 / 24 scenarios |

The evaluation also applied three fail-closed controls to every pair: remove
one target region, change one target class, and duplicate one same-class target
region. All 18 targeted controls abstained as required, with zero unexpected
links. Aggregate control precision was 100% and control recall was 62.5%.

## Where recall was lost

The benchmark records association eligibility independently from final
promotion. An expected identity is eligible only when the pair pose is
accepted and its exact region pair has at least three matches and three pose
inliers without a high region-pose residual.

| Evidence state | Identities | Promoted correctly |
| --- | ---: | ---: |
| Eligible association evidence | 8 | 8 |
| Relative pose rejected | 4 | 0 |
| Fewer than three regional inliers | 2 | 0 |

Thus conditional association recall was 100% (8/8). This is useful evidence
that the simple one-to-one associator behaves correctly inside its stated
operating envelope. It is not evidence that the full first-stage pipeline has
adequate recall: only 8 of 14 expected identities entered that envelope.

Two pairs explain the pose-coverage failure. `essential-stat-2055` returned a
degenerate, rejected pose. `essential-stat-2318` had many consensus inliers but
failed the explicit pose-quality policy, so the object pipeline correctly
refused to promote its two hypotheses. In `essential-stat-2275`, the pair pose
was accepted but both correct object pairs had only two regional matches and
one regional pose inlier.

## Interpretation

For V1, the evidence supports keeping the current conservative contract:

- create a unique object ID only for an accepted one-to-one region link;
- attach the scale-free spatial hypothesis produced from its supporting
  inliers;
- abstain when relative pose, semantic compatibility, regional support, or
  uniqueness is insufficient; and
- expose rejection reasons instead of forcing an identity.

The next recall work belongs at the feature/pose and regional-support seams,
not in a looser assignment threshold. Lowering the association gates would
trade the observed zero false promotions for unsupported IDs.

## Reproduction and provenance

The runner uses only RGB, frozen manual RGB rectangles, public spherical
features and matching, and the estimated relative pose during prediction.
Expected links are opened only after `prediction.json` is written read-only.
Stanford2D3D images remain external and are verified by exact SHA-256 hashes.

```bash
python benchmarks/object_localization/run_multipair_association_validation.py \
  --preflight

python benchmarks/object_localization/run_multipair_association_validation.py \
  --output-dir /private/tmp/panorai-object-association-multipair-final
```

The second command intentionally exits nonzero while the frozen end-to-end
gates fail. For the recorded run:

- prediction SHA-256:
  `15398da51c0bfbadaf78c20c076502198a82b06d8985ceb75134fd59637633c8`;
- reference SHA-256:
  `831d7ccc6667c934d4ff69bd976cd71e66760d021135c5ff6bdd9eea64ac8a91`.

This is dirty-source-checkout, post-hoc development evidence. It does not
validate automatic detector/CAM regions, graph persistence, dataset-wide
generalization, an installed wheel, or a release artifact.
