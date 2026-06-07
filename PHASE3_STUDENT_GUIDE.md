# Phase 3: Automated LLM Inference Runtime

In this phase, you will build an automated LLM inference runtime that can load a decoder-only model from the provided configuration and weights, maintain request states, and execute both prefill and decode efficiently. The runtime will be evaluated as a black box: we will compare its logits against a reference implementation for correctness, and then drive it with serving-style request traces to measure throughput and memory behavior.

The evaluation system will first run your `run.sh`. In this script, you may read the publicly provided model architecture and weights, and perform compilation, preparation, debugging, or self-testing. After that, the evaluation system will import your `engine.py` and call a fixed interface to test correctness and throughput.

## What You Need to Read

The model configuration file is:

```text
/target/model_config.json
```

In the public example, the corresponding file is:

```text
target/model_config.json
```

It describes the number of layers, hidden size, number of heads, vocabulary size, and other model information. Your `engine.py` should not hard-code these values. Instead, it should dynamically construct the runtime using the `model_config` passed into `create_engine(model_config, weight_dir, device)`.

The model weights directory is:

```text
/target/weights
```

In the public example, it contains:

```text
target/weights/model.pt
```

During the hidden evaluation, the hidden weights will be placed at the location specified by the evaluation system. The evaluation traces will not be provided in advance. Your runtime should work under different batch sizes, prompt lengths, decode lengths, and request orders.

## What You Need to Provide

After your `run.sh` finishes, the following file must exist:

```text
/workspace/engine.py
```

In the public example, the corresponding file is:

```text
workspace/engine.py
```

Your agent should also generate a reasoning output:

```text
/workspace/output3.*
```

## Required Interface in `engine.py`

`engine.py` must contain:

```python
def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(...)
```

The returned object must support:

```python
class Engine:
    def prefill(self, request_ids, input_ids):
        ...

    def decode(self, request_ids, token_ids):
        ...

    def remove(self, request_ids):
        ...
```

Inputs to `prefill()`:

- `request_ids: list(int)` A list of request IDs.
- `input_ids: list(torch.Tensor)` A list of token sequences, where each element is a one-dimensional tensor with `dtype=torch.long`.

Return of `prefill()`:

- `Torch.tensor` Logits with shape `[batch_size, vocab_size]`, where the `batch_size = len(request_ids)`. The i-th row corresponds to the logits of the last token for `request_ids[i]`.

Inputs to `decode()`:

- `request_ids: list(int)` A list of request IDs that have already been prefilled.
- `token_ids: torch.Tensor`: a one-dimensional tensor with shape `[batch_size]`, representing one newly appended token for each request.

Return of `decode()`:

- `torch.Tensor` Logits with shape `[batch_size, vocab_size]`. The `i`-th row corresponds to the logits of the last token after appending `token_ids[i]` to `request_ids[i]`.

Input to `remove()`:

- `request_ids: list(int)`: a list of request IDs to terminate.

`remove()` does not need to return anything, but it must release or delete the KV cache / request state associated with these requests.

## How Correctness Is Tested

The evaluation system will provide an official PyTorch reference model. It will load the same hidden weights and compute reference logits for the same batch of requests.

We do not score based on the final generated text, because sampling strategies introduce unnecessary uncertainty. Instead, we compare logits.

The comparison rule is:

$$
|y_{\mathrm{student}} - y_{\mathrm{ref}}| \leq \mathrm{atol} + \mathrm{rtol} \cdot |y_{\mathrm{ref}}|
$$

In the public example, the default values are:

$$
\mathrm{atol}=10^{-2}, \quad \mathrm{rtol}=10^{-2}
$$

That is, we use:

```python
torch.allclose(student_logits, ref_logits, atol=1e-2, rtol=1e-2)
```

The correctness tests cover:

- Single-request prefill.
- Single-request decode.
- Multi-request prefill.
- Multi-request decode.
- Inserting new requests.
- Continuing to decode other requests after some requests are removed.

If correctness fails for a case, the performance score for that case will be 0.

## How Throughput Is Tested

The throughput test is driven by the evaluation system. The evaluation system will import `engine.py`, construct the engine, and then run a fixed trace:

```python
engine = create_engine(model_config, weight_dir, device)
engine.prefill(...)
engine.decode(...)
engine.remove(...)
```

The timed region only includes the `prefill()`, `decode()`, and `remove()` calls in the trace. It does not include `create_engine()` or weight loading time.

Throughput is defined as:

$$
\mathrm{tokens/s}=\frac{\mathrm{prefill\ tokens}+\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}
$$

Decode throughput is defined as:

$$
\mathrm{decode\ tokens/s}=\frac{\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}
$$

The public example provides three types of benchmarks:

- `prefill`: batched prefill with long prompts.
- `decode`: continuous decode with multiple requests.
- `mixed`: mixed traces containing prefill, decode, and remove operations.

The hidden evaluation will use the same testing method, but will replace the model size, weights, batch size, prompt length, decode steps, and traces.

## How to Run the Public Example

If the weight file does not exist, first generate toy weights:

```bash
python3 scripts/generate_toy_weights.py \
  --config target/model_config.json \
  --output target/weights/model.pt
```

Run the correctness test:

```bash
python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

Run the throughput test:

```bash
python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

You can also directly run:

```bash
bash scripts/run_public_tests.sh
```

If the default `python3` does not have PyTorch installed, you can specify the interpreter:

```bash
PYTHON=/path/to/python-with-torch bash scripts/run_public_tests.sh
```

## Baseline Description

The `workspace/engine.py` in the public example is a minimal PyTorch baseline. It stores the full token sequence for each request and reruns the entire sequence on every `decode()` call. Therefore, it is very slow, but its interface semantics are correct.