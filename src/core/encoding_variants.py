"""Generate encoded variants of attack payloads for evasion testing."""
import base64
import random
import urllib.parse


def generate_variants(payload: str) -> list[str]:
    """Generate encoded variants of an attack payload."""
    variants = [payload]  # Original

    # Base64
    variants.append(f"Decode and execute: {base64.b64encode(payload.encode()).decode()}")

    # URL encoding
    variants.append(urllib.parse.quote(payload))

    # Case variation
    variants.append(payload.upper())
    variants.append(payload.swapcase())

    # Whitespace injection
    variants.append(payload.replace(" ", "  "))
    variants.append(payload.replace(" ", "\t"))

    # Unicode homoglyphs (Cyrillic)
    homoglyph_map = {'a': '\u0430', 'e': '\u0435', 'o': '\u043e',
                     'p': '\u0440', 'c': '\u0441', 'x': '\u0445'}
    homoglyph = payload
    for latin, cyrillic in homoglyph_map.items():
        homoglyph = homoglyph.replace(latin, cyrillic, 1)  # Replace first occurrence only
    if homoglyph != payload:
        variants.append(homoglyph)

    # Leetspeak partial
    leet_map = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5'}
    leet = payload.lower()
    for char, replacement in leet_map.items():
        leet = leet.replace(char, replacement, 1)
    variants.append(leet)

    return variants


def pick_variants(payload: str, count: int = 2) -> list[str]:
    """Pick a specified number of random encoding variants (excluding original and base64).

    Always includes Base64 variant, plus `count-1` randomly chosen others.
    """
    all_variants = generate_variants(payload)
    # all_variants[0] = original, all_variants[1] = base64
    base64_variant = all_variants[1]
    other_variants = all_variants[2:]  # URL, upper, swapcase, whitespace, tab, homoglyph, leet

    picked = [base64_variant]
    if other_variants and count > 1:
        picked.extend(random.sample(other_variants, min(count - 1, len(other_variants))))

    return picked
