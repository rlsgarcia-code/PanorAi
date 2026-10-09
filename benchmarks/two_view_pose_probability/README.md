# Two-view pose probability study

This directory defines the VAL-018 academic validation program for calibrated
acceptance and precision probabilities in PanorAi's spherical two-view R,t
pipeline.

The study keeps two models separate:

- capture probability from operator-controllable or operator-visible inputs;
- post-processing precision from frontend and estimator diagnostics.

See [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md) for the preregistered experiments,
outcomes, permitted predictors, leakage rules, validation design and academic
deliverables.

The next executable step is E0: inventory canonical Matterport360,
Stanford2D3D and P74 manifests at image, pair and group levels, then freeze
component-safe splits before any model is fitted.
