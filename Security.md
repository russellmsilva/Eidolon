# Security Model

Nosumina's harness asks an LLM to synthesize Python programs
("candidates") that model a game's rules, then executes those programs to
score them against recorded gameplay traces. This document describes how
untrusted, LLM-generated code is contained, and what the harness deliberately
does _not_ try to protect against.

## Threat model

**Untrusted:** every candidate program the LLM produces, regardless of intent.
The harness does not distinguish between "the model made an honest mistake,"
"the model hallucinated something dangerous," and "the model was manipulated
via adversarial input into producing something dangerous" — all three are
handled by the same containment, because there is no reliable way to tell
them apart from the outside. Nothing about _why_ a candidate looks the way it
does changes how much it's trusted.

**Trusted:** the harness/orchestrator code itself, the host environment, and
game-trace data used to build prompts (this is the operator's own recorded
data, not third-party input).

**Explicitly out of scope:**

- Host kernel vulnerabilities that would let a process escape OS-level
  namespace isolation entirely. This class of sandbox is not a hypervisor;
  it shares the host kernel by design.
- Supply-chain integrity of dependencies (the model weights, Python
  packages, the sandboxing tool itself).
- Multi-tenant isolation between unrelated users — this harness is designed
  for a single operator running their own candidates, not as a service that
  executes code on behalf of untrusted third parties.

## Defense layers

Containment is layered, and no single layer is assumed to be sufficient on
its own:

1. **Static allowlist check.** Candidate source is parsed (not executed) and
   checked against an explicit allowlist of importable modules before it's
   ever staged for execution. This is a fast-fail for the common case, not a
   security boundary — it only inspects `import` statements and can't see a
   candidate reaching for the same capability through a builtin function
   instead.

2. **Restricted execution namespace.** The candidate's code executes with a
   reduced set of builtins available to it — file I/O and dynamic
   code-execution builtins are removed, and the module-import mechanism
   itself is wrapped to enforce the same allowlist as (1) at runtime, closing
   the gap that (1) alone leaves open. This is a courtesy filter for
   careless or lightly-adversarial code, not a hard guarantee: a
   sufficiently determined program could still reach live objects already
   resident in the interpreter through language-level introspection that
   never touches the restricted names at all. The real boundary is (3).

3. **OS-level sandboxing.** Every candidate execution runs inside a
   kernel-namespace sandbox (via `bubblewrap`) with:
   - No network access
   - A read-only filesystem view, with no writable location provided
     anywhere in the sandbox
   - All Linux capabilities dropped
   - Process/PID isolation, so a killed sandbox process cannot leave
     orphaned children behind
   - No shared controlling terminal with the host process
   - A synthetic, minimal `/proc` and `/dev`, not bind-mounts of the host's

   This is the actual enforcement boundary. Layers 1 and 2 exist to fail
   fast on the common case and reduce noise; they are not relied upon to
   contain a determined adversary.

4. **Resource limits.** CPU time, memory, process count, wall-clock
   duration, and total captured output are all bounded, independent of
   whatever the candidate's own code tries to do — including cases where a
   candidate manages to interfere with an in-process timeout mechanism, since
   the outermost time limit is enforced from outside the sandboxed process
   entirely.

5. **Structural output validation.** Values a candidate returns are
   validated for basic shape/type correctness on the host side before being
   used downstream, so a malformed or unexpected return value fails as a
   clean, contained result rather than surfacing as an unrelated-looking
   crash later in the pipeline.

## Known limitations

- **Kernel-shared execution.** Namespace-based sandboxing constrains what a
  process can _reach_, not what the underlying kernel itself is vulnerable
  to. A kernel-level privilege-escalation exploit is not something this
  layer of defense can stop; the mitigation is keeping the host kernel
  current, not anything in this codebase.
- **Python-level restrictions are best-effort.** Restricting names in an
  execution namespace does not remove the underlying objects from the
  process — a sufficiently determined program with knowledge of Python
  internals can, in principle, route around it. This is a known, accepted
  limitation of any restricted-builtins approach in a language with this
  much runtime introspection, and it's why layer 3 exists as the actual
  boundary rather than layers 1–2.
- **Shared address space with the inference engine (local-model backend
  only).** When running a local model in-process rather than via a separate
  inference server, the orchestrator and the model's inference code share
  one process. An isolated, server-based inference backend is available and
  can be used if this exposure ever becomes more relevant to the operator's
  threat model than it currently is.

## Responsible disclosure

If you find a way to break out of the sandbox, bypass a defense layer, or otherwise get a candidate program to do something this document says it shouldn't be able to do, please report it privately rather than opening a public issue or PR — a public report gives anyone reading the repo a working exploit before there's a fix.

- **Email:** russell.miguel.silva [at] gmail [dot] com. Include what you found, how to reproduce it, and (if you have one) how bad the impact is — a candidate reading files it shouldn't vs. a full sandbox escape are very different severities and it helps to say which you think this is.
- There's no bug bounty here; this is a research project, not a funded product. If that changes later, this section will say so.

This threat model treats every candidate program as untrusted regardless of intent (see above) — so a "the model just happened to write something exploitable, nobody was actually attacking it" report is exactly as welcome and useful as a deliberately crafted one. Both are real gaps in the same containment.

## Testing

Sandbox containment is exercised by an automated smoke test "sandbox_smoke_test.py"
that drives the real sandbox path with synthetic inputs and disposable toy programs
designed to attempt each of the bypasses layers 1–4 are meant to catch. It's meant to
be re-run after any change to the sandbox configuration or the set of
allowed imports.
