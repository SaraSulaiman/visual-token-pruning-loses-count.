# When Visual Token Pruning Loses Count: evaluation items and records

This folder gives the exact items the paper tests on, how they were chosen, how answers are read, and the counts behind the
reported percentages. It holds lists and records only: no images, no model outputs, no model weights.

## 1. The items (exact IDs)

| Set | Size | File | Fields |
|---|---|---|---|
| Photographs, COCO val2017 | 227 photos with 3-6 instances of one category (78 / 76 / 49 / 24 photos with 3 / 4 / 5 / 6) | `items/photos_227.json` | COCO file name, question, true count |
| Robot-view frames, OCID | 1,870 frames with 0-10 objects, from all 89 scenes, each seen by 2 cameras (178 sequences) | `items/ocid_1870.json` | path inside the OCID download, scene, sequence, camera, question, true count |
| VQAv2 | 5,208 questions on 1,000 val2014 images | `items/vqav2_5208.json` | VQAv2 question id, COCO file name, question |

## 2. The questions (prompts)

- Photographs: "How many <category, plural> are there?", for example "How many people are there?". The exact text is in the item file.
- OCID: "How many objects are on the table?" or "How many objects are on the floor?", by scene type.
- VQAv2: the original question.
- Wrapper per model: LLaVA-1.5 and Prismatic append "Answer the question using a single word or phrase." inside their own chat format.
  PaliGemma uses "answer en <question>". All decoding is greedy.

## 3. How the items were chosen, and what was left out

**Photographs.** A fixed rule was applied to the COCO val2017 annotations before any pruned model was run:

- one target category per image whose non-crowd instance count is 3 to 6, with no crowd annotation of that category
- every target instance at least 32 x 32 pixels and at least 0.1 % of the image; the target instances together at least 2 % of the image
- 19 categories excluded: bed, book, broccoli, carrot, couch, dining table, hair drier, keyboard, laptop, microwave, mouse, oven, refrigerator, scissors, sink, skis, toaster, toilet, tv
- 722 photos qualified; they were visited in a seeded random order (seed 2027) and taken unless a cap was full: 40 photos per category,
  90 per count, 100 per clutter level. This drew 260 photos.
- A person then checked every drawn photo against its annotation, and a photo was kept only if both its count and its masks were confirmed.
  33 were removed: 22 were marked wrong on both, and 11 were not confirmed. **227 remain.** `photos_removed_33.json` lists the 33 removed photos
  with the two check results (`false` = marked wrong, `null` = not confirmed).

**OCID.** All 89 scenes of the dataset, both cameras. The true count is the number of object ids in the label mask. Frames with more than
10 objects were dropped (520 frames). Nothing else was removed.

**VQAv2.** 1,000 val2014 images drawn with a fixed random seed (NumPy generator, seed 0) from a larger fixed pool of validation images,
with every question of those images (5,208). The exact list is in the item file, so the draw does not have to be repeated.
Scored with the official VQAv2 evaluator.

## 4. How an answer is read

`answer_parser.py`: the first integer in the answer; if there is none, the first number word (zero to twelve). An answer with neither is
counted as wrong.

## 5. OCID count ranges and pairs

A pair is two consecutive frames of one sequence whose counts differ by exactly one. The frame list gives 1,692 such pairs.
The count range of a model family is every count at which its all-token model answers at least half the frames correctly.

| Family (reference model) | Count range | Frames in range | Pairs in range |
|---|---|---|---|
| LLaVA-1.5, 576 tokens | 0-3 | 624 | 446 |
| PaliGemma 448 px, 1024 tokens | 0-5 | 980 | 802 |
| Prismatic 224 px, 256 tokens | 0-3 | 624 | 446 |
| Prismatic 384 px, 729 tokens | 0-3 | 624 | 446 |

**Are the pairs identical across models?** Within a family, yes: every pruned or cut arm is scored on exactly the frames and pairs of its
family's range. Across families, only when the ranges agree: LLaVA-1.5 and both Prismatic models share the same 446 pairs; PaliGemma's
range is wider and adds the pairs with 4 and 5 objects.

## 6. How many items stand behind each percentage

Every "kept" share and break rate is a share of the items the all-token model of that family answers correctly. Every repair rate is a
share of the remaining items.

| Reference model | Photographs right (of 227) | OCID frames right (of the frames in range) | VQAv2 answers right (of 5,208) | VQAv2 number questions right (of 652) |
|---|---|---|---|---|
| LLaVA-1.5, 576 tokens | 77 | 470 of 624 | 3,518 | 328 |
| PaliGemma 448 px, 1024 tokens | 136 | 716 of 980 | 3,940 | 443 |
| PaliGemma 224 px, 256 tokens | 132 | not used | 3,909 | 410 |
| Prismatic 224 px, 256 tokens | 106 | 513 of 624 | 3,643 | 352 |
| Prismatic 384 px, 729 tokens | 113 | 543 of 624 | 3,725 | 372 |

"Exactly +1" and "both frames right" are shares of all pairs in range (section 5). A VQAv2 answer counts as right at full credit.

Robot policy: the all-token policy solves **424 of 500** episodes (84 of the 100 episodes used for the smaller comparisons).
Probe: with all tokens the probe counts **758 of 1,870** frames correctly.

## 7. The probe: scenes used for training and testing

1,870 OCID frames from 89 scenes, split into five folds by scene, so that no scene is used for both training and testing.
Each scene is tested by a probe trained on the other four folds. `probe_scene_folds.json` gives the test fold of every scene.

## 8. Tests fixed in advance

| Test | Record | Written |
|---|---|---|
| LLaVA-1.5 on OCID | `preregistrations/llava_ocid_2026-10-03.json` | 2026-10-03 15:12 |
| PaliGemma position control | `preregistrations/paligemma_positions_2026-10-07.json` | 2026-10-07 14:19 |
| Prismatic 384 px against 224 px | `preregistrations/prismatic_384_vs_224_2026-10-07.json` | 2026-10-07 14:19 |

Each record states what was fixed before the run: the questions, token budgets, arms, scoring, and the rule for passing.
The paper's other analyses were not fixed in advance.
