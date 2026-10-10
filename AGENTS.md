# Longbook27 — agent instructions

## Purpose and scope

Build a small, local GGUF inference client for
`summerMC/Qwen3.8-27B-Longbook27-VL-GDN64-GGUF`.
The first implementation is a native C++ CLI, with Linux/CPU verification and
optional CUDA builds. Keep the engine interface usable by a future GUI.
User instructions in the active session take precedence over this document.

## Work order

1. Read this file and `docs/PR_POLICY.md` before changing code.
2. Inspect the pinned model and engine facts in `models/longbook27.json` and
   `engine.lock.json` when those files exist; never infer compatibility from a name.
3. Make one focused change per branch and explain its purpose and resulting behavior.
4. Build and run the relevant checks; state what was actually verified.
5. Update the documentation whenever behavior or supported scope changes.

This file was created before implementation. Local bootstrap work may create the
initial documentation commit before feature branches exist.

## Engine and model invariants

- Use the model author's `vnlpscale/llama.cpp` compatibility fork at a full,
  immutable commit SHA. Do not silently replace it with upstream `master`.
- This checkpoint has 64 recurrent Gated DeltaNet layers. Retain the recurrent
  graph, state handling, GGUF loader, tokenizer, sampler, and required GGML kernels.
- Prune by an explicit source allowlist and focused patches. Build target removal
  and actual source removal must be reported separately.
- Preserve all upstream license and attribution notices in redistributed code.
- Keep model weights outside version control and release source archives.
- Validate GGUF metadata before allocating the full model. A compatible architecture
  is not proof that the weights are this particular checkpoint.
- Keep native context, verified context, and speculative extensions distinct.
  Never describe a theoretical or model-card experiment as a verified client feature.
- Do not claim image support from the `VL` name alone. Enable it only with an
  identified, compatible projector, implementation, and validation evidence.
- Never silently truncate a document or shift/drop recurrent state to make it fit.
- Model loading and inference stay local. Network access must be an explicit
  provisioning/download action, never an implicit chat side effect.

## Implementation conventions

- C++17 or newer, RAII for resource ownership, explicit error handling.
- CMake is the build entry point; Python 3 standard library is preferred for tools.
- Keep user text on stdout and diagnostics on stderr. Return nonzero on failures.
- Validate integers, finite floats, paths, context limits, and model/projection inputs.
- Pass argument vectors to subprocesses; do not construct executable shell strings
  from user input. Do not embed tokens, credentials, or machine-specific paths.
- Keep the client small: no HTTP server, telemetry, converter, training code, or
  unrelated model architectures in the initial runtime distribution.
- Write code comments in clear English; user-facing project documentation may use
  Japanese. Avoid unsupported performance, memory, and compatibility promises.

## Verification

- Check a clean configure/build and the native CLI help/error paths.
- Exercise behavior affected by the change, especially token budgets, batch boundaries,
  recurrent decoding, prompt formatting, EOF/interrupts, and failure cleanup.
- Keep fast tests independent of large model downloads. Use clearly labeled synthetic
  fixtures only for structural checks, never as evidence of model quality.
- Actual-checkpoint claims require the model revision, file checksum, engine SHA,
  hardware, quantization, context, prompt, and observed result.
- CUDA/Metal/Vulkan results must be marked unverified unless tested on that backend.

## PR discipline

Follow `docs/PR_POLICY.md` and `.github/pull_request_template.md`.
Engine pin updates, topology changes, pruning changes, and measured performance
claims need reproducible evidence and separate reviewable changes.
Do not disable a failing gate, rewrite unrelated history, or claim review/CI that
has not occurred. Local commits and preparation are authorized by implementation
work; publishing, repository administration, and contacting others follow the
user's actual instructions rather than assumptions in this file.
