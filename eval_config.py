"""
eval_config.py
==============
Which question ids are reserved as prompt examples (stage 3/4), which ids make
up the fixed 20-question dev subset used for cheap smoke tests, and the stage names.

evaluation_dataset_v2.csv has 100 rows, contiguous by complexity:
Easy = ids 1-30, Medium = ids 31-70, Hard = ids 71-100 (30/40/30 split).
"""

# Ids used as worked examples inside the stage3 (one-shot) and stage4
# (few-shot) prompts. Reserved so the model is never scored on a question
# it was already shown the answer to.
#   4  -> aggregation / top-N            (Easy)
#   14 -> simple filter                  (Easy)
#   22 -> combined boolean filter        (Easy)
#   63 -> date-range filter + HAVING     (Medium)
RESERVED_FEWSHOT_IDS = [4, 14, 22, 63]

# Fixed 20-question dev subset, stratified proportionally to the real
# 30/40/30 Easy/Medium/Hard split (-> 6/8/6), picked by even spacing within
# each contiguous id block while skipping the reserved ids above.
DEV_SUBSET_IDS = [
    2, 7, 12, 17, 24, 27,             # Easy   (6 of 30)
    34, 39, 44, 49, 54, 59, 64, 69,   # Medium (8 of 40)
    73, 78, 83, 88, 93, 98,           # Hard   (6 of 30)
]

assert not set(DEV_SUBSET_IDS) & set(RESERVED_FEWSHOT_IDS), \
    "Dev subset must not overlap with reserved few-shot ids"

STAGES = ["stage1_simple", "stage2_rules", "stage3_one_shot", "stage4_few_shot"]
