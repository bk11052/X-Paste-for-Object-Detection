"""
Expand scenario seeds in configs/scenarios.yaml into N detailed SDXL prompts.

Phase 1 (manual web mode): prints the prompt to copy into ChatGPT, accepts the
pasted JSON response, validates and caches it. Already-cached scenarios are
skipped on rerun, so progress is incremental.

Future: add --backend openai|gemini for automated API generation. The output
JSON schema is shared across modes so downstream scripts (gen_singleshot_scenes.py)
do not need to change.

Usage:
    python generation/gen_scenario_prompts.py \
        --scenarios configs/scenarios.yaml \
        --output generation/cache/scenario_prompts.json \
        [--force]                # regenerate all (overwrites cache)
        [--only 1,3,9]           # only these scenario ids
        [--n_prompts 4]          # how many prompts per scenario
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


SYSTEM_PROMPT = """You are an expert prompt engineer for Stable Diffusion XL (SDXL).

Given a short scene seed, expand it into N detailed and DIVERSE prompts that SDXL will use to generate photorealistic MILITARY BACKGROUND images.

Hard constraints for every prompt:
- The scene MUST be PHOTOREALISTIC, war-photography / documentary style.
- The scene MUST NOT contain any soldiers, people, faces, vehicles, tanks, cars, trucks, helicopters, planes, weapons, or military equipment. Objects will be pasted in later by a separate pipeline.
- Each prompt is 30-60 words.
- Each prompt should specify: camera / lens (e.g. wide shot, telephoto, 35mm), lighting, time of day, weather, atmosphere, and ground/horizon detail.
- The N prompts should DIFFER from each other in lighting, weather, time of day, viewpoint, or surface detail, while keeping the same scenario context.
- Use English. Avoid named real people, copyrighted franchises, or text/logos.

Output: a single JSON array of exactly N strings. NO prose, NO markdown, NO trailing comma. Just the JSON array.
"""


def build_user_prompt(seed: str, n_prompts: int) -> str:
    return (
        f"N = {n_prompts}\n"
        f"Seed: {seed}\n\n"
        f"Return the JSON array of {n_prompts} prompts now."
    )


BANNED_TOKENS = [
    "soldier", "soldiers", "people", "person", "human", "man", "woman", "men", "women",
    "tank", "tanks", "vehicle", "vehicles", "car", "cars", "truck", "trucks",
    "helicopter", "plane", "aircraft", "weapon", "weapons", "rifle", "gun",
]


def validate_prompts(prompts: list, n_expected: int) -> tuple[bool, str]:
    if not isinstance(prompts, list):
        return False, "response is not a JSON array"
    if len(prompts) != n_expected:
        return False, f"expected {n_expected} prompts, got {len(prompts)}"
    for i, p in enumerate(prompts):
        if not isinstance(p, str):
            return False, f"prompt #{i} is not a string"
        if len(p) < 20:
            return False, f"prompt #{i} too short ({len(p)} chars)"
        low = p.lower()
        hits = [t for t in BANNED_TOKENS if t in low.split()]
        if hits:
            return False, f"prompt #{i} contains banned tokens {hits}: {p[:80]}..."
    return True, ""


def read_pasted_json(stdin) -> str:
    print(
        "\n>>> Paste ChatGPT's JSON response below. End with a line containing only 'EOF':\n",
        flush=True,
    )
    lines = []
    for line in stdin:
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "".join(lines).strip()


def parse_json_response(raw: str) -> list:
    raw = raw.strip()
    # tolerate fenced code blocks
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    return json.loads(raw)


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True, help="path to configs/scenarios.yaml")
    ap.add_argument("--output", required=True, help="path to JSON cache to write/update")
    ap.add_argument("--force", action="store_true", help="regenerate even if already cached")
    ap.add_argument("--only", default="", help="comma-separated scenario ids to process")
    ap.add_argument("--n_prompts", type=int, default=4, help="number of expanded prompts per scenario (default 4)")
    args = ap.parse_args()

    scenarios_path = Path(args.scenarios)
    output_path = Path(args.output)

    cfg = yaml.safe_load(scenarios_path.read_text())
    scenarios = cfg["scenarios"]
    shared_negative = cfg.get("global", {}).get("shared_negative", "").strip()

    only_ids = set()
    if args.only:
        only_ids = {int(x) for x in args.only.split(",") if x.strip()}

    cache = load_cache(output_path)
    todo = []
    for s in scenarios:
        sid = str(s["id"])
        if only_ids and s["id"] not in only_ids:
            continue
        if sid in cache and not args.force:
            print(f"[skip] {sid} {s['name']} (cached)")
            continue
        todo.append(s)

    if not todo:
        print("Nothing to do. Use --force to regenerate.")
        return 0

    print(
        f"\n=== {len(todo)} scenario(s) to process ===\n"
        f"For each scenario:\n"
        f"  1. Copy the SYSTEM + USER block into a fresh ChatGPT chat.\n"
        f"  2. Paste ChatGPT's JSON array back here.\n"
        f"  3. End with a line containing only 'EOF' on its own.\n"
        f"  Type 'SKIP' (then EOF) to skip, 'QUIT' to stop.\n"
    )

    for s in todo:
        sid = str(s["id"])
        print("\n" + "=" * 78)
        print(f"Scenario {sid}: {s['name']}  ({s.get('title_kr', '')})")
        print("=" * 78)
        print("\n----- COPY BELOW THIS LINE INTO CHATGPT -----")
        print(f"[SYSTEM]\n{SYSTEM_PROMPT}")
        print(f"[USER]\n{build_user_prompt(s['background_prompt_seed'], args.n_prompts)}")
        print("----- COPY ABOVE THIS LINE -----")

        while True:
            raw = read_pasted_json(sys.stdin)
            if raw.strip() == "QUIT":
                save_cache(output_path, cache)
                print("Stopped. Cache saved.")
                return 0
            if raw.strip() == "SKIP":
                print(f"[skip] {sid}")
                break
            try:
                prompts = parse_json_response(raw)
            except json.JSONDecodeError as e:
                print(f"  ! JSON parse error: {e}. Retry.")
                continue
            ok, why = validate_prompts(prompts, args.n_prompts)
            if not ok:
                print(f"  ! validation failed: {why}. Retry.")
                continue

            cache[sid] = {
                "name": s["name"],
                "title_kr": s.get("title_kr", ""),
                "seed": s["background_prompt_seed"],
                "prompts": prompts,
                "negative_prompt": " ".join(
                    x for x in [shared_negative, s.get("background_negative", "")] if x
                ),
                "instances_spec": s["instances"],
                "n_backgrounds": s.get("n_backgrounds", args.n_prompts * 4),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "ChatGPT (manual web)",
            }
            save_cache(output_path, cache)
            print(f"  [ok] saved {args.n_prompts} prompts for scenario {sid}")
            break

    print(f"\nAll done. Cache: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
