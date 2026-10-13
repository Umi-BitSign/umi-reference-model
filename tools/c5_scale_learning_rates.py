"""Explicit visual/decoder optimizer groups for two matched C5 ablations."""

from __future__ import annotations

SPLIT_RATES = {
    "c5-openasl-scale-fused-lora-split-visual-fast-v1": (1e-4, 4e-6),
    "c5-openasl-scale-fused-lora-split-decoder-fast-v1": (4e-6, 1e-4),
}


def split_recipe(common: dict, recipe_id: str) -> dict:
    visual_rate, decoder_rate = SPLIT_RATES[recipe_id]
    recipe = {key: value for key, value in common.items() if key != "learning_rate"}
    return {
        **recipe, "id": recipe_id,
        "visual_learning_rate": visual_rate, "decoder_learning_rate": decoder_rate,
    }


def optimizer_groups(trainable: dict, recipe: dict) -> list[dict]:
    from c5_retention import ALL_SOURCE_IDS
    if recipe["id"] not in SPLIT_RATES and recipe["id"] not in ALL_SOURCE_IDS:
        return [{"name": "all_lora", "params": list(trainable.values()), "lr": recipe["learning_rate"]}]
    if recipe["id"] in ALL_SOURCE_IDS:
        from c5_all_source_training import RECIPE
        expected = (RECIPE["visual_learning_rate"], RECIPE["decoder_learning_rate"])
    else:
        expected = SPLIT_RATES[recipe["id"]]
    if (recipe.get("visual_learning_rate"), recipe.get("decoder_learning_rate")) != expected:
        raise ValueError("frozen visual/decoder learning rates differ")
    visual = [parameter for name, parameter in trainable.items() if "signhubert" in name]
    decoder = [parameter for name, parameter in trainable.items() if name.startswith("decoder.")]
    if not visual or not decoder or len(visual) + len(decoder) != len(trainable):
        raise ValueError("split optimizer lacks an exact visual/decoder partition")
    if len({id(parameter) for parameter in (*visual, *decoder)}) != len(trainable):
        raise ValueError("split optimizer repeats a parameter")
    return [
        {"name": "visual_lora", "params": visual, "lr": expected[0]},
        {"name": "decoder_lora", "params": decoder, "lr": expected[1]},
    ]
