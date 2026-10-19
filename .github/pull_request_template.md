## Public contract

- [ ] This change has no new public surface, or every new/changed surface is registered in `contracts/public-api-surfaces-v2.json`.
- [ ] I selected the smallest correct role: Geometry, typed data/result, spherical operation, Torch layer, estimator/policy, Pipeline Stage, Application, subsystem, or compatibility surface.
- [ ] Coordinate frame, units, grid, support, validity, interpolation and optional-dependency behavior are explicit where applicable.

## Documentation

- [ ] The documentation-role assignment routes the surface to a catalog, API reference, concept and progressive guide.
- [ ] Application examples are registered; every Stable Application has a source example and an installed-wheel consumer.
- [ ] The README policy is deliberate: `featured` for an entry-point capability, otherwise `catalog-only`.
- [ ] Strict Sphinx, documentation-contract tests and executable examples pass.

## Evidence

- [ ] Tests use real public objects and include an independent oracle where correctness is claimed.
- [ ] Source, wheel, sdist, installed-wheel and external evidence are named accurately.
- [ ] Optional dependencies and third-party licenses remain explicit and lazy.

See the [public API component contract](../docs/development/application_extensions.rst).
