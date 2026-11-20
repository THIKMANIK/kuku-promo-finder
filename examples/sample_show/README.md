# Sample show

Synthetic test data written for this repo, not Kuku content.

Two short episodes (about 70 seconds each) of an invented story, "The Flower Shop", in the same format as
the screenplay input pack: header lines, `SCENE` blocks, and `**TYPE (emotion):**` blocks with
`Start: (mm:ss.xx) | End: (mm:ss.xx)` lines. They are used by the tests and for trying the tool without an API key:

```bash
python run.py --input examples/sample_show --provider mock
```
