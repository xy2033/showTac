#!/usr/bin/env python3
"""
Generate tactile QA pairs from contact_indoor_list_tvl.csv.

The QA pairs are designed for Show-O2 Stage 2 NTP training, following the
same chat-template NTP pattern as the original MMUDataset:
  - Question tokens → text_labels = -100 (ignored)
  - Answer tokens   → text_labels = valid token IDs (NTP loss computed)

Output columns:
  object_name, question, answer, qa_type

Reference:
  VTV-LLM / TouchThinker 四维静态属性（本数据集精简到可用部分）:
    Hardness:   soft / medium / hard
    Roughness:  smooth / slightly rough / rough
    Material:   plastic / metal / rubber / glass / wood / ceramic / cardboard / foam / paper
    Texture:    glossy / matte / metallic / fibrous / sleek / polished / waxy

Changelog v2 (fixes from Codex review):
  1. material_discrimination: randomize option order (was always correct=first)
  2. texture_tag: negation-aware matching (non-porous ≠ porous)
  3. material: cross-reference object_name hints when text has conflicts
  4. hardness: match "hardness level is low/high" patterns; conservative default
  5. roughness: "low roughness" treated as smooth indicator, not slightly_rough
"""

import csv
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Attribute extraction from free-text descriptions
# ---------------------------------------------------------------------------

# Material keyword hints from object_name (for cross-referencing conflicts)
_MATERIAL_HINTS = {
    'plastic': ['plastic'],
    'metal': ['metal', 'iron', 'aluminum', 'steel', 'copper', 'can_metal'],
    'rubber': ['rubber', 'rubber_', 'silicone'],
    'glass': ['glass', 'galss'],  # typo in dataset: galss
    'wood': ['wood'],
    'ceramic': ['ceramic'],
    'cardboard': ['cardboard'],
    'foam': ['foam', 'sponge'],
    'paper': ['paper'],
}


def _material_from_name(object_name: str) -> Optional[str]:
    """Try to infer material from object_name as a hint."""
    name_lower = object_name.lower()
    for mat, hints in _MATERIAL_HINTS.items():
        for hint in hints:
            if hint in name_lower:
                return mat
    return None


def extract_material(full_sentence: str, object_name: str = '') -> str:
    """Extract material from 'it is made of X' pattern, with object_name hint."""
    text_material = 'unknown'
    match = re.search(r'it is made of (\w+)', full_sentence, re.IGNORECASE)
    if match:
        mat = match.group(1).lower()
        if mat in ('plastic', 'metal', 'rubber', 'glass', 'wood', 'ceramic',
                    'cardboard', 'foam', 'paper'):
            text_material = mat
        elif any(t in mat for t in ['plastic']):
            text_material = 'plastic'
        elif any(t in mat for t in ['metal', 'aluminum', 'iron', 'steel']):
            text_material = 'metal'
        elif any(t in mat for t in ['rubber', 'silicone']):
            text_material = 'rubber'

    # Cross-reference with object_name hint for conflict resolution
    name_hint = _material_from_name(object_name)
    if name_hint and text_material != 'unknown' and name_hint != text_material:
        # Conflict: text says X, name suggests Y → prefer text unless
        # name is very explicit (e.g. object_name starts with material_)
        # We trust object_name for clearly-named objects, text for others
        if object_name.startswith(name_hint + '_') or object_name.startswith(name_hint):
            return name_hint  # override: object name is more reliable
    if text_material != 'unknown':
        return text_material
    if name_hint:
        return name_hint
    return 'unknown'


def extract_roughness(full_sentence: str) -> Tuple[str, str]:
    """
    Extract roughness level.
    Returns (level, description) where level is one of:
      smooth, slightly_rough, rough

    Fix v2: 'low roughness' = smooth indicator (not slightly_rough).
             'fine texture' alone does not override stronger smooth signals.
    """
    text = full_sentence.lower()

    # --- Tier 1: Strong smooth signals ---
    strong_smooth = ['very smooth', 'completely smooth', 'extremely smooth',
                     'no roughness', 'lacks any noticeable texture',
                     'frictionless', 'almost frictionless',
                     'low roughness level', 'very low roughness',
                     'low roughness']
    if any(t in text for t in strong_smooth):
        return 'smooth', 'smooth'

    # --- Tier 2: Strong rough signals ---
    strong_rough = ['high roughness', 'very rough', 'abrasive',
                    'rough and hard surface', 'distinctly uneven and bumpy']
    if any(t in text for t in strong_rough):
        return 'rough', 'rough'

    # --- Tier 3: Moderate / slightly rough signals ---
    moderate_rough = ['slightly coarse', 'slightly rough',
                      'moderate roughness', 'medium roughness',
                      'some roughness', 'moderately high roughness']
    if any(t in text for t in moderate_rough):
        return 'slightly_rough', 'slightly rough'

    # --- Tier 4: Weak signals — score-based ---
    weak_rough = ['light texture', 'fine texture', 'minimal roughness',
                  'slightly rough surface']
    weak_smooth = ['smooth', 'polished', 'sleek', 'flat surface',
                   'even texture', 'uniform texture', 'glossy finish']

    rough_score = sum(1 for t in weak_rough if t in text)
    smooth_score = sum(1 for t in weak_smooth if t in text)

    # Also check word-boundary 'rough' vs 'smooth'
    if _word_in_text('rough', text):
        rough_score += 1
    if _word_in_text('coarse', text):
        rough_score += 1
    if 'smooth' in text:
        smooth_score += 1

    if rough_score > smooth_score:
        return 'slightly_rough', 'slightly rough'
    elif smooth_score > rough_score:
        return 'smooth', 'smooth'
    else:
        return 'smooth', 'smooth'  # default conservative


def _word_in_text(word: str, text: str) -> bool:
    """Check if a word/phrase appears as a whole-word match in text."""
    # For multi-word phrases, simple substring is fine
    if ' ' in word or len(word) > 6:
        return word in text
    # For single short words, use word boundary
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


def extract_hardness(full_sentence: str) -> Tuple[str, str]:
    """
    Extract hardness level.
    Returns (level, description).

    Fix v2: Match "hardness level is low/high" and "hardness is low/high" patterns.
            Use word-boundary matching to avoid 'hard' in 'hardness' false positive.
    """
    text = full_sentence.lower()

    # --- Tier 1: Explicit hardness level patterns ---
    # Match "hardness level is low", "hardness is low", "hardness level is high", etc.
    low_hardness_patterns = [
        r'hardness\s+(?:level\s+)?is\s+low', r'hardness\s+(?:level\s+)?is\s+very\s+low',
        r'hardness\s+(?:level\s+)?is\s+relatively\s+low', r'low\s+hardness',
        r'hardness\s+(?:is\s+)?soft',
    ]
    high_hardness_patterns = [
        r'hardness\s+(?:level\s+)?is\s+high', r'hardness\s+(?:level\s+)?is\s+very\s+high',
        r'high\s+hardness', r'high\s+level\s+of\s+hardness',
    ]
    medium_hardness_patterns = [
        r'medium\s+hardness', r'moderate\s+hardness',
        r'some\s+hardness',
    ]

    for pat in low_hardness_patterns:
        if re.search(pat, text):
            return 'soft', 'soft'
    for pat in high_hardness_patterns:
        if re.search(pat, text):
            return 'hard', 'very hard'
    for pat in medium_hardness_patterns:
        if re.search(pat, text):
            return 'medium', 'medium hardness'

    # --- Tier 2: Very explicit signals ---
    if _word_in_text('very soft', text):
        return 'soft', 'very soft'
    if _word_in_text('very hard', text) or _word_in_text('extremely hard', text):
        return 'hard', 'very hard'

    # --- Tier 3: Score-based with word-boundary matching ---
    soft_indicators = ['soft', 'compressible', 'spongy', 'yielding',
                        'flexible rubber', 'akin to soft', 'pliable',
                        'low in hardness']
    hard_indicators = ['firm', 'rigid', 'unyielding', 'inflexible', 'solid',
                        'hard plastic', 'highly rigid', 'hard rubber']

    soft_score = sum(1 for t in soft_indicators if _word_in_text(t, text))
    hard_score = sum(1 for t in hard_indicators if _word_in_text(t, text))
    # Only count 'hard' with word boundary (avoids 'hardness' false positive)
    if _word_in_text('hard', text):
        hard_score += 1

    if soft_score > hard_score:
        return 'soft', 'soft'
    elif hard_score > soft_score:
        return 'hard', 'hard'
    else:
        # Conservative: when truly uncertain, check if any soft indicator at all
        if soft_score > 0:
            return 'soft', 'soft'
        return 'medium', 'medium hardness'  # changed from hard → medium


def _is_negated(kw: str, text: str) -> bool:
    """Check if a keyword is negated in the text (e.g. 'non-porous' negates 'porous')."""
    # Common negation prefixes
    for prefix in ['non-', 'not ', 'no ']:
        if prefix + kw in text:
            return True
    # "lacks any noticeable texture" negates texture-like words
    if 'lacks any noticeable texture' in text and kw in ('textured', 'texture'):
        return True
    # "no noticeable roughness" negates roughness-like words
    if 'no noticeable roughness' in text and kw == 'rough':
        return True
    # "no apparent roughness"
    if 'no apparent roughness' in text and kw == 'rough':
        return True
    # "no texture"
    if 'no texture' in text and kw in ('textured', 'rough'):
        return True
    return False


def extract_texture_tags(full_sentence: str) -> List[str]:
    """Extract texture adjectives present in the text. Negation-aware."""
    text = full_sentence.lower()
    tags = []

    # Keywords grouped by category
    texture_keywords = [
        'glossy', 'matte', 'waxy', 'sleek', 'reflective', 'polished',
        'fibrous', 'shiny', 'metallic', 'slick', 'slippery',
        'spongy', 'textured', 'ridged', 'bumpy',
        'frictionless', 'flat', 'even',
        # porous checked separately with negation
    ]

    for kw in texture_keywords:
        if kw in text and not _is_negated(kw, text):
            tags.append(kw)

    # Handle 'porous' specially — frequently negated as 'non-porous'
    if 'porous' in text and not _is_negated('porous', text):
        tags.append('porous')

    # 'smooth' is safe (no common negation in this dataset)
    if 'smooth' in text:
        tags.append('smooth')

    return tags


def extract_flexibility(full_sentence: str) -> Optional[str]:
    """Extract flexibility characteristic."""
    text = full_sentence.lower()
    if any(t in text for t in ['flexible', 'elastic', 'slightly flexible',
                                 'pliable', 'yielding under pressure']):
        return 'flexible'
    if any(t in text for t in ['rigid', 'inflexible', 'unyielding', 'stiff',
                                 'solid', 'firm and durable']):
        return 'rigid'
    return None


def extract_temperature(full_sentence: str) -> Optional[str]:
    """Extract temperature characteristic."""
    text = full_sentence.lower()
    if any(t in text for t in ['cool to the touch', 'cool temperature',
                                 'cold to touch', 'cold sensation',
                                 'cool and crisp', 'coolness']):
        return 'cool'
    return None


def extract_object_display_name(object_name: str, full_sentence: str) -> str:
    """Extract a human-readable object display name from full_sentence."""
    # full_sentence usually starts with "The touch of X is ..."
    match = re.match(r'The touch of (.+?) is', full_sentence, re.IGNORECASE)
    if match:
        name = match.group(1).strip().rstrip('.').rstrip(',')
        # Remove trailing descriptors
        name = re.sub(r'\s+with\s+.*$', '', name)
        return name
    return object_name.replace('_', ' ')


# ---------------------------------------------------------------------------
# QA pair generators
# ---------------------------------------------------------------------------

def gen_object_name_qa(obj_name: str, display_name: str, material: str) -> List[Dict]:
    """Type A: Simple object/material identity QA."""
    qa_pairs = []

    # A1: What object is being touched?
    qa_pairs.append({
        'question': 'What object is being touched?',
        'answer': f'The object being touched is a {display_name}.',
        'qa_type': 'object_identity',
    })

    # A2: What material?
    if material != 'unknown':
        qa_pairs.append({
            'question': 'What material is the object made of?',
            'answer': f'The object is made of {material}.',
            'qa_type': 'material_identity',
        })

    # A3: Binary material discrimination — randomized option order
    other_materials = ['plastic', 'metal', 'rubber', 'glass', 'wood', 'ceramic']
    if material in other_materials:
        other_materials.remove(material)
    if other_materials:
        distractor = random.choice(other_materials)
        # Fix v2: randomize option order to prevent position shortcut
        options = [material, distractor]
        random.shuffle(options)
        qa_pairs.append({
            'question': f'Is this object made of {options[0]} or {options[1]}?',
            'answer': f'This object is made of {material}.',
            'qa_type': 'material_discrimination',
        })

    return qa_pairs


def gen_roughness_qa(roughness_level: str, roughness_desc: str,
                     material: str) -> List[Dict]:
    """Type B: Roughness attribute QA."""
    qa_pairs = []

    # B1: Direct roughness question
    qa_pairs.append({
        'question': 'Is the surface of this object smooth or rough?',
        'answer': f'The surface is {roughness_desc}.',
        'qa_type': 'roughness_binary',
    })

    # B2: Material-specific roughness
    if material != 'unknown':
        qa_pairs.append({
            'question': f'Based on the tactile feedback, does this {material} object feel smooth?',
            'answer': f'{"Yes" if "smooth" in roughness_level else "No"}, the {material} object feels {roughness_desc}.',
            'qa_type': 'roughness_material',
        })

    return qa_pairs


def gen_hardness_qa(hardness_level: str, hardness_desc: str,
                    material: str) -> List[Dict]:
    """Type C: Hardness attribute QA."""
    qa_pairs = []

    # C1: Direct hardness question — adapt answer for medium
    if hardness_level == 'medium':
        qa_pairs.append({
            'question': 'Is this object hard or soft?',
            'answer': f'The object has {hardness_desc}, neither distinctly hard nor soft.',
            'qa_type': 'hardness_binary',
        })
    else:
        qa_pairs.append({
            'question': 'Is this object hard or soft?',
            'answer': f'The object is {hardness_desc}.',
            'qa_type': 'hardness_binary',
        })

    # C2: Material-specific hardness
    if material != 'unknown':
        if hardness_level == 'hard' or hardness_level == 'very_hard':
            feel_answer = f'Yes, the {material} object feels {hardness_desc}.'
        elif hardness_level == 'medium':
            feel_answer = f'The {material} object has {hardness_desc}, neither very hard nor soft.'
        else:
            feel_answer = f'No, the {material} object feels {hardness_desc}.'
        qa_pairs.append({
            'question': f'Does this {material} object feel hard when pressed?',
            'answer': feel_answer,
            'qa_type': 'hardness_material',
        })

    return qa_pairs


def gen_combined_qa(roughness_desc: str, hardness_desc: str,
                    material: str, texture_tags: List[str]) -> List[Dict]:
    """Type D: Combined 2-attribute QA."""
    qa_pairs = []

    # D1: Roughness + hardness
    qa_pairs.append({
        'question': 'What are the roughness and hardness of this object?',
        'answer': f'The surface is {roughness_desc} and the object feels {hardness_desc}.',
        'qa_type': 'roughness_hardness',
    })

    # D2: Material + hardness
    if material != 'unknown':
        qa_pairs.append({
            'question': f'What material is used and how hard does it feel?',
            'answer': f'The object is made of {material} and feels {hardness_desc}.',
            'qa_type': 'material_hardness',
        })

    # D3: Texture tag if available
    if texture_tags:
        tag_str = ', '.join(texture_tags[:3])
        qa_pairs.append({
            'question': 'What texture can be observed on the surface?',
            'answer': f'The surface has a {tag_str} texture.',
            'qa_type': 'texture_tag',
        })

    return qa_pairs


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    csv_path = '/home/xy/showO/Show-o/show-o2/contact_indoor_list_tvl.csv'
    output_path = '/home/xy/showO/Show-o/docs/idea/tactile_qa_pairs.csv'

    # Read source CSV
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue
            obj_name = row[0].strip()
            full_sentence = row[-3].strip().strip("'\"")
            samples.append((obj_name, full_sentence))

    print(f"Loaded {len(samples)} objects from CSV")

    # Generate QA pairs
    all_qa_pairs = []
    stats = Counter()

    for obj_name, full_sentence in samples:
        # Extract attributes
        material = extract_material(full_sentence, obj_name)
        roughness_level, roughness_desc = extract_roughness(full_sentence)
        hardness_level, hardness_desc = extract_hardness(full_sentence)
        texture_tags = extract_texture_tags(full_sentence)
        display_name = extract_object_display_name(obj_name, full_sentence)
        flexibility = extract_flexibility(full_sentence)
        temperature = extract_temperature(full_sentence)

        # Generate QA pairs for this object
        qas = []

        # Type A: Object identity (2-3 QAs per object)
        qas.extend(gen_object_name_qa(obj_name, display_name, material))

        # Type B: Roughness (2 QAs)
        qas.extend(gen_roughness_qa(roughness_level, roughness_desc, material))

        # Type C: Hardness (2 QAs)
        qas.extend(gen_hardness_qa(hardness_level, hardness_desc, material))

        # Type D: Combined (2-3 QAs)
        qas.extend(gen_combined_qa(roughness_desc, hardness_desc, material, texture_tags))

        # Add object_name to each QA and collect
        for qa in qas:
            qa['object_name'] = obj_name
            stats[qa['qa_type']] += 1

        all_qa_pairs.extend(qas)

    # Shuffle to mix QA types
    random.shuffle(all_qa_pairs)

    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['object_name', 'question', 'answer', 'qa_type'])
        for qa in all_qa_pairs:
            writer.writerow([
                qa['object_name'],
                qa['question'],
                qa['answer'],
                qa['qa_type'],
            ])

    print(f"\nGenerated {len(all_qa_pairs)} QA pairs → {output_path}")
    print(f"\nQA type distribution:")
    for qtype, count in stats.most_common():
        print(f"  {qtype}: {count}")

    # Print some examples
    print("\n=== Example QA pairs ===")
    for qa in all_qa_pairs[:10]:
        print(f"\n[{qa['qa_type']}] {qa['object_name']}")
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")


if __name__ == '__main__':
    main()
