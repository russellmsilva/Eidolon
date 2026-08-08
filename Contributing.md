# Contributing to Eidolon

Eidolon is a program synthesis harness for ARC-AGI-3: a local LLM writes and revises a `GameModel` class that predicts how a game's grid changes from one step to the next, and that code gets checked by replaying it against everything recorded so far. The core idea, inspired by the Schema paper (Zeng et al. 2026), is that how you scaffold a model matters more than how big the model is. The longer-term goal is for that idea to hold up outside ARC-AGI-3 too — the hope is that a harness like this could eventually be adapted to other domains where a model needs to build and revise a working model of its environment from observation, like robotics or medicine, not just games. More broadly, the aim is to show that safe, well-scaffolded automation built on local models — not just ever-larger frontier ones — can be a real path toward post-scarcity. That vision only holds if the automation getting there is safe and legible, not a black box — which is why this project treats glass-box design (sandboxed execution of anything the model writes, and every decision, limitation, and rejected approach written down rather than hidden) as a goal in its own right, not an afterthought.

A couple of things Eidolon adds on top of what Schema's paper describes: Eidolon is built to run entirely on a local, quantized model with no internet access, rather than assuming access to a frontier model — a real constraint Schema's approach doesn't have to work within. It also adds a sandboxing layer (`bwrap`, a restricted set of allowed imports) for safely running code the model writes, since that code needs to actually execute untrusted.

Every contribution gets judged against one simple question: does this help the model reason better, or does it end up doing the model's reasoning for it? Good scaffolding makes the model's job easier without taking over the parts that are supposed to be its job — deciding what a shape means, what the rule is, what counts as progress. If a change starts making those decisions instead of the model, that's usually a sign it's solving the wrong problem.

## New here?

A few things worth knowing before you dive in:

- `SECURITY.md` explains the sandboxing setup and why it exists.
- The "Known Limitations" section at the end of each implementation plan is a running list of things that were tried and didn't pan out, or gaps that are still open. It's worth a skim before proposing something new — there's a decent chance an idea has already been tried, and knowing why it didn't work will save you some time. We've already been through a few rounds of "this looked right until we checked it against real data" on some of these, so there are probably more problems we haven't found yet either — that's normal, not a sign anything's broken.

## Contributor agreement

Contributions here are volunteer. There's no equity, compensation, or other consideration implied or promised for contributing code, tests, docs, or anything else to this repo — Eidolon is, and is intended to stay, open source under MIT. If you're contributing with an expectation of future equity, a paid role, or a cofounder-type relationship if this ever turns into a company, say so and let's have that conversation explicitly and separately — don't assume it's implied by the act of contributing.

This project uses the **Developer Certificate of Origin (DCO)** instead of a copyright-assignment CLA. Practically, that means:

- Every commit needs a `Signed-off-by` line, added automatically with `git commit -s`. This certifies you wrote the contribution (or otherwise have the right to submit it under the project's license) — see [`DCO.txt`](./DCO.txt) (mirrored from [developercertificate.org](https://developercertificate.org)) for the exact text you're agreeing to. This automatically appends your signature to the bottom of the commit message: `Signed-off-by: John Doe <john.doe@example.com>`
- You keep copyright on your own contributions. Signing off licenses your contribution to the project (and everyone downstream) under MIT — it doesn't transfer ownership. This is deliberate: no single party, including a future company built by anyone involved with this project, can ever take the whole codebase private without going back to get every contributor's consent first.
- If you have an employer, it's worth actually checking your employment agreement's IP-assignment clause before contributing, especially if your work relates in any way to what your employer does. Many employment contracts claim rights to work created during employment — including personal-time work, in some cases — and being open source doesn't exempt a contribution from that if it was never yours to give away in the first place. A few states have carve-outs for personal projects unrelated to your employer's business, but the details vary and "unrelated" is doing real work in that sentence. This is worth five minutes of your own attention, not something the DCO checkbox verifies for you.

## Trying something new

Small fixes, tests, docs, and extensions to something that already exists: normal pull request, no extra process needed.

If you're proposing something bigger — a new component, a different way of representing state, a new synthesis strategy — it helps everyone (including you) to test the idea small before building it out fully. Open an issue with:

- what problem it solves, ideally tied to something we've actually seen go wrong
- whether this has been tried before (check the Known Limitations sections)
- a small test that would tell us whether the idea holds up, before a big build
- what you'd expect to see if it didn't work, not just if it did

Then run that small test and share what happened before starting the full build. This isn't about gatekeeping ideas — it's the same thing we do with our own ideas before committing real time to them, and it means nobody spends a week building something that a half-day test would have already told us not to.

## The sandbox is load-bearing

The sandbox (`bwrap`, a restricted set of allowed imports, output limits, a grid-shape validator) exists because the code running through this harness is written by a model, and needs to be treated as untrusted even when nothing's wrong with it. It's easy to want to loosen something here just to get a feature working faster — please don't do that quietly. If a guardrail is genuinely getting in the way, open a discussion about changing it, rather than routing around it in a branch. Changes here get an extra look before merging, not because we don't trust contributors, but because this is the one place where a small mistake has outsized consequences.

## Code conventions

- CI runs pure-Python logic only (parsing, validation, template rendering). Anything that needs a real LLM call or the actual sandbox stays a manual test — CI can't exercise those anyway.
- Comments and docstrings should make sense on their own, without needing to look up a design doc or step number. Explain the "why" inline.
- Code is the source of truth. If something in an older design doc doesn't match what's actually there, trust the code — and feel free to send a small PR fixing the doc.

## Review

PRs touching the sandbox get a closer look, same reasoning as above. For a big new direction proposed without a small test first, we'll probably ask to see that test before diving into full review — not a rejection, just sequencing.
