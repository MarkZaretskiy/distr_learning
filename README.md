### Added new methods for model_dc_framework

transferring to gpu/cpu

`.to("cuda")/.to("cpu")` or `.cuda()/.cpu()`

loading model from hard disk

`.load("path/to/checkpoint")`

validation

`.val(data)`

### How to test

`cd 0.\ Prod\ DL/task_framework/tests/`

`python3 test_simple.py`