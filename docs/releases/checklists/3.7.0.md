# PanorAi 3.7.0 release checklist

This checklist records the gates for the exact candidate commit. It does not
authorize tagging or publication.

- [ ] Quality Gates 0 and A–I pass for the exact commit.
- [ ] Python 3.11, 3.12, 3.13, and 3.14 test matrices are green.
- [ ] Strict Sphinx build passes.
- [ ] API inventory validates against its JSON Schema and covers every audited export.
- [ ] Stable and Compatibility imports pass installed-wheel tests.
- [ ] `panorai.graph` imports without optional perception backends.
- [ ] JSONL/NPZ replay and corruption tests pass from the installed wheel.
- [ ] Wheel/sdist content and license audits pass.
- [ ] No private dataset, checkpoint, report, or operational history is packaged.
- [ ] Tagging and publication receive explicit human authorization.
