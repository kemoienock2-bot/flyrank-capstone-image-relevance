# Build Log

Honest record of where Claude helped while building this, where it got
things wrong or out of date, and what I changed because of it. This is a
draft — I'm filling it in as I actually built things, so some of this is
written in Claude's voice describing what happened; I've gone through and
adjusted anything that didn't match how it actually felt to build.

## Design phase (Phase 1)

Claude walked me through each schema decision one at a time instead of
handing me a finished spec — category enum vs. free text, the `other`/
`uncertain` split, caption format, confidence structure, database shape.
I made the actual calls (closed enum, split confidence, no `attributes`
field for v1); Claude's job was laying out the tradeoffs and explaining
*why* a choice mattered before I picked. This part worked well and I
don't have much to correct here — it matched how I wanted to work.

## Where AI got things wrong or out of date

**Model names.** The first version of the vision-tagging code used
`gemini-1.5-flash` via the `google.generativeai` package. Both were fully
dead by the time I actually ran the code — the model returned a 404, and
the package itself is deprecated. This wasn't a small tweak; the whole
API client pattern changed (new `google-genai` package, `Client()` object
instead of `GenerativeModel()`). We had to debug this live, with me
pasting real error messages back and forth until it was fixed. This is
worth remembering: AI's knowledge of specific product names, SDKs, and
model IDs goes stale fast in a space that moves this quickly, and it
can't always know that on its own — it took real error output from
actually running the code to catch it.

**The `gemini-flash-latest` alias resolved to a bad choice.** After
fixing the SDK, we switched to an alias meant to always point at "the
current stable Flash model" — except it resolved to a *preview* model
with a 20-requests/day quota, which I burned through almost immediately.
Had to pin to a specific stable model name instead. Lesson: an alias that
sounds safe (`latest`) isn't automatically the right choice for a
free-tier project where quota matters more than having the newest model.

**Google Cloud project quota was more restrictive than documented.**
Even the "stable" model I switched to was capped at 20 requests/day on my
account — well below the couple-hundred/day the public docs suggested.
Took two separate rounds of hitting `429` errors and reading the actual
quota error messages (which included the real numbers) to figure out this
wasn't a code bug, just an account-tier limit. Ended up running the batch
job across two separate days to get through all 44 images.

## Where a design assumption was wrong, and got caught by real data

The similarity threshold approach almost had a real bug baked into it. The
first pass at matching (pure caption-embedding similarity, no category
check) showed a wolf photo scoring *higher* against a fox blog post than
against an actual wolf post. If I'd picked a single global similarity
threshold without looking closely at this, the system would have
approved a wrong-species match in production. Claude caught this by
actually printing out the full unfiltered similarity distribution before
suggesting any threshold — which is the part I'd flag as genuinely useful
AI-assisted work here: not writing the fix, but insisting on looking at
real numbers before committing to a threshold, which is what surfaced the
problem in the first place. The fix (an explicit category-match guard
step, checked before similarity) was then straightforward once the
problem was visible.

## What I changed / would do differently

- I'd ask to see the raw, unfiltered similarity numbers earlier in the
  process, before any threshold gets picked — it's what caught the real
  bug, and I almost skipped straight to picking a "reasonable-sounding"
  threshold.
- I'd double check model/SDK names against current docs *before* writing
  the integration code, not after hitting a 404 — that's on me as much as
  on the tool.
- Everything in `EVIDENCE.md` about calibration limits (small sample,
  hand-picked thresholds) is something I want to be upfront about at the
  demo, not something to gloss over.

## What I own

The actual design decisions — the schema shape, the guard's decision
order, what counts as a good enough match — were mine to make, with
Claude laying out tradeoffs rather than deciding for me. The code itself
was AI-assisted throughout, tested against real API responses and real
data as we went rather than trusted blind. Every number in `EVIDENCE.md`
came from actually running the pipeline against real images and real API
calls, not simulated or invented.
