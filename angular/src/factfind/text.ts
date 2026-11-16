/**
 * Text normalisation and fuzzy similarity.
 *
 * These mirror the Python side exactly: `normaliseLabel` is
 * `knowledge.normalise_label`, and `ratio` / `tokenSetRatio` reproduce
 * rapidfuzz's `fuzz.ratio` and `fuzz.token_set_ratio` on the inputs this
 * pipeline gives them. Keeping them equivalent is what lets the exported
 * model see the same `label_score` in the browser as in training.
 */

/** Lower-case, strip accents and punctuation, collapse whitespace. */
export function normaliseLabel(text: unknown): string {
  if (text === null || text === undefined) return '';
  return String(text)
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function despace(s: string): string {
  return s.replace(/ /g, '');
}

/** Levenshtein distance, iterative with a single row. */
export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;

  let previous = new Array<number>(b.length + 1);
  let current = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j++) previous[j] = j;

  for (let i = 1; i <= a.length; i++) {
    current[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= b.length; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      current[j] = Math.min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost);
    }
    const swap = previous;
    previous = current;
    current = swap;
  }
  return previous[b.length];
}

/**
 * rapidfuzz `fuzz.ratio`: the indel similarity, 0..100.
 *
 * rapidfuzz uses the Indel distance (Levenshtein without substitution), which
 * is what `ratio` is defined on. Substitution costs 2 there, so we compute it
 * from the longest common subsequence.
 */
export function ratio(a: string, b: string): number {
  if (!a.length && !b.length) return 100;
  if (!a.length || !b.length) return 0;
  const lcs = longestCommonSubsequence(a, b);
  return (2 * lcs * 100) / (a.length + b.length);
}

function longestCommonSubsequence(a: string, b: string): number {
  let previous = new Array<number>(b.length + 1).fill(0);
  let current = new Array<number>(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const ca = a.charCodeAt(i - 1);
    current[0] = 0;
    for (let j = 1; j <= b.length; j++) {
      current[j] = ca === b.charCodeAt(j - 1)
        ? previous[j - 1] + 1
        : Math.max(previous[j], current[j - 1]);
    }
    const swap = previous;
    previous = current;
    current = swap;
  }
  return previous[b.length];
}

function tokens(s: string): string[] {
  return s.split(/\s+/).filter(Boolean);
}

/**
 * rapidfuzz `fuzz.token_set_ratio`: compares the shared tokens against each
 * side's remainder, so "date of birth" and "birth date" score 100.
 */
export function tokenSetRatio(a: string, b: string): number {
  const ta = new Set(tokens(a));
  const tb = new Set(tokens(b));
  if (!ta.size && !tb.size) return 100;

  const intersection = [...ta].filter((t) => tb.has(t)).sort();
  const onlyA = [...ta].filter((t) => !tb.has(t)).sort();
  const onlyB = [...tb].filter((t) => !ta.has(t)).sort();

  const shared = intersection.join(' ');
  const combinedA = (shared + ' ' + onlyA.join(' ')).trim();
  const combinedB = (shared + ' ' + onlyB.join(' ')).trim();

  if (shared.length && (!onlyA.length || !onlyB.length)) return 100;
  return Math.max(
    ratio(shared, combinedA),
    ratio(shared, combinedB),
    ratio(combinedA, combinedB),
  );
}

/** rapidfuzz `fuzz.partial_ratio`: best ratio of `needle` against any window. */
export function partialRatio(needle: string, haystack: string): number {
  if (!needle.length || !haystack.length) return 0;
  if (needle.length > haystack.length) return partialRatio(haystack, needle);
  let best = 0;
  for (let i = 0; i + needle.length <= haystack.length; i++) {
    best = Math.max(best, ratio(needle, haystack.substr(i, needle.length)));
    if (best === 100) break;
  }
  return best;
}

/**
 * rapidfuzz `fuzz.WRatio`: ratio, token-set and a length-adjusted partial,
 * weighted the way rapidfuzz weights them.
 */
export function wRatio(a: string, b: string): number {
  if (!a.length || !b.length) return 0;
  const base = ratio(a, b);
  const lengthRatio = Math.max(a.length, b.length) / Math.min(a.length, b.length);

  if (lengthRatio < 1.5) {
    return Math.max(base, tokenSetRatio(a, b) * 0.95);
  }
  const partialScale = lengthRatio > 8 ? 0.6 : 0.9;
  return Math.max(
    base,
    partialRatio(a, b) * partialScale,
    tokenSetRatio(a, b) * 0.95 * partialScale,
  );
}

/** Best match from a list, as [value, score 0..100, index]. */
export function extractOne(
  query: string,
  choices: string[],
  scorer: (a: string, b: string) => number = wRatio,
): [string, number, number] | null {
  let best: [string, number, number] | null = null;
  for (let i = 0; i < choices.length; i++) {
    const score = scorer(query, choices[i]);
    if (!best || score > best[1]) best = [choices[i], score, i];
  }
  return best;
}

/** Top-n matches, best first. */
export function extract(
  query: string,
  choices: string[],
  limit = 5,
  scorer: (a: string, b: string) => number = wRatio,
): Array<[string, number, number]> {
  const scored = choices.map((c, i) => [c, scorer(query, c), i] as [string, number, number]);
  scored.sort((x, y) => y[1] - x[1]);
  return scored.slice(0, limit);
}

/** Title-case a name, respecting McDonald, O'Brien, van der Berg, hyphens. */
const NAME_PARTICLES = new Set([
  'de', 'la', 'van', 'von', 'der', 'den', 'del', 'di', 'da',
  'al', 'bin', 'ibn', 'le', 'dos', 'das',
]);

export function titlecaseName(value: string): string {
  const out: string[] = [];
  for (const word of value.split(/\s+/).filter(Boolean)) {
    const low = word.toLowerCase();
    if (NAME_PARTICLES.has(low) && out.length) {
      out.push(low);
      continue;
    }
    out.push(word.split('-').map(capToken).join('-'));
  }
  return out.join(' ');
}

function capToken(token: string): string {
  if (!token) return token;
  const low = token.toLowerCase();
  if (low.startsWith('mc') && low.length > 2) {
    return 'Mc' + low.charAt(2).toUpperCase() + low.slice(3);
  }
  if (low.startsWith("o'") && low.length > 2) {
    return "O'" + low.charAt(2).toUpperCase() + low.slice(3);
  }
  const apostrophe = low.indexOf("'");
  if (apostrophe > 0) {
    const a = low.slice(0, apostrophe);
    const b = low.slice(apostrophe + 1);
    return cap(a) + "'" + cap(b);
  }
  return cap(low);
}

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}
