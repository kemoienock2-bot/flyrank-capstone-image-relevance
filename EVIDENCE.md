# Evidence & Results

This documents what actually happened when the pipeline ran against real
data — the numbers, the mistakes it caught, and a few things that went
wrong along the way (and how they got fixed).

## Dataset

43 images across 4 categories, collected from Unsplash with source URL and
license logged at download time:

| Category | Count |
|---|---|
| fox | 13 |
| wolf | 13 |
| dog | 13 |
| other (cats/birds, deliberately out of scope) | 5 |

One image (out of an original 44) was rejected during tagging — its caption
came back at 27 words against a 10–25 word limit. It was left rejected, not
manually fixed, since silently patching model output would hide a real
signal about when the model struggles.

## Tagging accuracy

Comparing what Gemini predicted against the ground-truth category baked
into the file path (`data/raw/{category}/...`):

**42 / 43 correct — 97.7% accuracy**

Confusion matrix:

| Ground truth | dog | fox | other | wolf |
|---|---|---|---|---|
| dog | 13 | 0 | 0 | 0 |
| fox | 0 | 13 | 0 | 0 |
| other | 0 | 0 | 5 | 0 |
| wolf | 0 | 0 | 1 | 11 |

The one miss: a real wolf photo (`wolf/0006.jpg`) got tagged as `other`
instead of `wolf`. Worth noting this isn't a pipeline bug — it's the vision
model itself making a judgment call on one specific photo. The tagging
schema's separation between "other" and "uncertain" meant this came through
as a clean, single-category mistake rather than something messier.

## Matching coverage

Of the 43 tagged images, how many ended up with at least one approved match:

**36 / 43 images had at least one approved match — 83.7%**

But that number on its own is misleading, so here's the breakdown:

| Category | Approved coverage |
|---|---|
| fox | 13/13 (100%) |
| wolf | 11/11 (100%) |
| dog | 12/13 (92%) |
| other | 0/6 (0%) |

The `other` category shows 0% on purpose — the guard rejects `other`-tagged
images before similarity is even checked, since there's no legitimate blog
post for a cat or bird photo in this dataset. That's the guard working
correctly, not a failure.

Removing `other` from the picture: **36 / 37 in-scope images matched — 97.3%
effective coverage.**

The one real gap: `dog/0011.jpg` was tagged correctly as a dog, but its
caption didn't score high enough against any of the three dog-related blog
posts (best score was 0.15, threshold was 0.20). This is the guard doing
exactly what it's meant to do — refusing to force a weak match rather than
picking the least-bad option. It's a content gap, not a pipeline error.

## A real problem the guard caught mid-build

While tuning the similarity thresholds, a genuine flaw showed up in the raw
similarity scores: a wolf photo scored *higher* against a fox blog post
(0.4858) than against an actual wolf blog post (0.4315). Pure text
similarity was picking up on descriptive writing style — the fox posts used
a lot of physical, visual language that happened to overlap with any
snow-animal caption — rather than actual species identity.

A similarity score alone couldn't have caught this reliably. The fix was
adding an explicit category-match check to the guard: reject any pairing
where the image's category doesn't match the blog post's expected category,
*before* similarity is even computed. After that fix, every remaining
mismatch in the "rejected before similarity check" bucket is a real category
mismatch, not a false positive slipping through on wording overlap.

This is documented here because it's a good example of the project's core
idea working as intended: a guard that only looked at one signal (similarity)
would have made a real mistake. Checking multiple independent signals in
sequence caught it.

## Guard rejection breakdown (all 344 image x post pairs)

| Reason | Count |
|---|---|
| Category mismatch (image category ≠ post's expected category) | 209 |
| Category out of scope ("other") | 48 |
| Below similarity threshold | 14 |
| **Approved** | **73** |

Every rejected pair has a specific, stored reason — nothing is rejected
with a generic "no."

## Threshold calibration — an honest limitation

Similarity thresholds were set per category based on the real score
distributions observed in this dataset, not learned from a large labeled
set:

| Category | Threshold |
|---|---|
| fox | 0.40 |
| wolf | 0.25 |
| dog | 0.20 |

These numbers came from eyeballing where legitimate matches clustered
against 8 blog posts (1–3 posts per category). That's a small sample. A
production version of this system would want a much larger, properly
labeled evaluation set before trusting these exact cutoffs — this dataset
was enough to prove the *approach* of per-category thresholds is necessary
(a single global threshold clearly didn't work, as shown above), but not
enough to fully trust the specific numbers.

## What broke during the build, and what it taught

A few real problems came up while actually running this against live APIs
— worth documenting since they shaped some of the design:

- **Model names go stale fast.** `gemini-1.5-flash` was fully retired
  between when this was planned and when it was run. Switched to the
  `gemini-flash-latest` alias, then had to pin back to a specific stable
  model (`gemini-2.5-flash`) once the alias resolved to a preview model
  with a much stricter daily quota (20 requests/day vs. hundreds for
  stable models).
- **Free-tier quotas are per-model, not just per-account.** Hitting a quota
  wall on one model name and switching to another can reset the count,
  since Google tracks quota separately per model.
- **This is exactly why the retry/no-retry split mattered.** Transient
  errors (429s, a 503) got retried automatically and mostly recovered.
  Non-transient errors (a fully retired model, a bad request) failed fast
  instead of wasting retries on something retrying couldn't fix.

## Summary

- 97.7% tagging accuracy against ground-truth folder labels
- 97.3% effective matching coverage on in-scope categories
- One real vision-model misclassification, caught cleanly by the
  category schema
- One real embedding-similarity flaw, caught and fixed via an added
  guard step, with the fix documented and testable
- Every rejection across 344 scored pairs has a specific, human-readable
  reason stored in the database
