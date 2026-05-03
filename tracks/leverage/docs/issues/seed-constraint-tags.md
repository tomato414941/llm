# Seed Constraint Tags

Status: proposed
Date: 2026-05-03

## Background

The current reviewed dataset is biased toward short assistant answers. The
`Qwen/Qwen3.5-9B` LoRA adapter then regressed on IFEval examples that require
long answers while preserving surface constraints such as length, punctuation,
case, forbidden words, and JSON keys.

The current seed schema can express those requirements with `constraints`, but
the values are free-form strings.

## Problem

Free-form `constraints` are useful for teacher generation and judging, but they
are weak as planning metadata:

- constraint families are hard to count reliably
- long-form constraints can be hidden inside natural-language strings
- multi-constraint examples are hard to distinguish from simple exact-answer
  examples
- data growth can accidentally add more short exact-answer rows while appearing
  to cover `instruction_following`

## Proposal

Consider adding an optional seed-only field:

```json
"constraint_tags": ["long_min_words", "no_comma", "multi_constraint"]
```

Use `constraint_tags` only for planning, auditing, and distribution checks. Keep
the natural-language `constraints` field as the source contract for teacher
generation and judging.

## Non-Goals

- Do not add new `capability` values for surface constraints.
- Do not add this field to reviewed rows yet.
- Do not create IFEval-specific tags or copy IFEval prompts.
- Do not make every seed require tags until the value is proven.

## Adoption Criteria

Adopt `constraint_tags` only if the next long-form surface-constraint seed batch
is hard to plan or audit with the existing `constraints` field alone.

The first useful evidence would be a seed batch where tag counts would have
prevented repeated short exact-answer examples or made long-form coverage
clearer.

## Initial Tag Candidates

- `long_min_words`
- `no_comma`
- `no_punctuation`
- `case_constraint`
- `forbidden_word`
- `json_key_value_constraint`
- `multi_constraint`
