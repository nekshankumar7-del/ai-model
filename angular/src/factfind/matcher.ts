/**
 * Segmentation and field matching - the browser twin of segment.py and
 * matcher.py.
 *
 * Segmentation turns each OCR line into the plausible ways to read it as a
 * label and a value. Matching scores every field in the spec against those
 * readings, using the field's labels, whether the value parses under its
 * kind, and which section heading is in force.
 */
import {
  FactFindSpec, FieldSpec, OcrLine, ParseResult, SectionSpec,
} from './spec.types';
import { ValueParser } from './parsers';
import { despace, normaliseLabel, ratio, tokenSetRatio } from './text';

export interface Pair {
  labelText: string;
  valueText: string;
  separator: string;
  confidence: number;
}

export interface LineCandidate {
  line: OcrLine;
  sectionId: string;
  isHeading: boolean;
  pairs: Pair[];
  prevText: string;
  nextText: string;
}

export interface FieldMatch {
  key: string;                 // qualified key, 'personal-details.fullName'
  field: FieldSpec;
  pair: Pair;
  labelScore: number;
  labelMargin: number;
  parse: ParseResult;
  sectionBonus: number;
  score: number;
  linkerScore: number;
  line: LineCandidate;
}

const SEPARATORS: Array<[RegExp, string]> = [
  [/\s*[:：]\s*/, 'colon'],
  [/\s*\.{3,}\s*/, 'leader'],
  [/\s*[-–—]\s+/, 'dash'],
  [/\s*[=>]\s*/, 'arrow'],
  [/\s{3,}/, 'gap'],
];

const NARRATIVE_SPLIT = /\s*[,;]\s+|\s+and\s+|\s*\.\s+(?=[a-zA-Z])/;

export class Matcher {
  private readonly parser: ValueParser;
  private readonly defaultSection: SectionSpec;
  private readonly headingIndex: Array<[string, string]>;

  constructor(private readonly spec: FactFindSpec) {
    this.parser = new ValueParser(spec);
    this.defaultSection =
      spec.sections.find((s) => s.default) ?? spec.sections[0];
    const pairs: Array<[string, string]> = [];
    for (const section of spec.sections) {
      for (const heading of section.headings) {
        pairs.push([normaliseLabel(heading), section.id]);
      }
    }
    for (const heading of spec.otherHeadings) {
      pairs.push([normaliseLabel(heading), '__other__']);
    }
    this.headingIndex = pairs
      .filter(([stem]) => stem.length > 0)
      .sort((a, b) => b[0].length - a[0].length);
  }

  get valueParser(): ValueParser {
    return this.parser;
  }

  get defaultSectionId(): string {
    return this.defaultSection.id;
  }

  // ------------------------------------------------------------ headings
  private detectSection(text: string, current: string): [string, boolean] {
    const norm = normaliseLabel(text);
    if (!norm || norm.length > 60) return [current, false];

    const colon = /\s*[:：]\s*/.exec(text);
    const hasValue = !!colon && colon.index + colon[0].length < text.trim().length;
    const words = norm.split(' ');
    if (words.length > 6) return [current, false];

    for (const [stem, sectionId] of this.headingIndex) {
      const hit = words.some(
        (w) => w.startsWith(stem) || (stem.startsWith(w) && w.length >= 4),
      ) || norm.startsWith(stem);
      if (!hit) continue;
      // "income" as a heading vs "Income: 45000" as a field line.
      if (hasValue && words.length <= 2) return [current, false];
      return [sectionId, true];
    }
    return [current, false];
  }

  // -------------------------------------------------------- segmentation
  private splitOnce(text: string): Pair[] {
    for (const [pattern, name] of SEPARATORS) {
      const m = pattern.exec(text);
      if (!m) continue;
      const labelText = text.slice(0, m.index).trim();
      const valueText = text.slice(m.index + m[0].length).trim();
      if (!labelText) continue;
      let confidence = valueText ? 1 : 0.4;
      if (name === 'gap') confidence *= 0.75;
      if (name === 'dash') confidence *= 0.9;
      return [{ labelText, valueText, separator: name, confidence }];
    }
    return [];
  }

  private splitByWords(text: string): Pair[] {
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length < 2) return [];
    const out: Pair[] = [];
    const maxCut = Math.min(words.length, this.spec.segmentation.maxLabelWords + 1);
    for (let cut = 1; cut < maxCut; cut++) {
      const labelText = words.slice(0, cut).join(' ');
      const valueText = words.slice(cut).join(' ');
      if (!labelText || !valueText) continue;
      out.push({ labelText, valueText, separator: 'words',
        confidence: 0.6 - 0.06 * (cut - 1) });
    }
    return out;
  }

  private splitNarrative(text: string): Pair[] {
    const chunks = text.split(NARRATIVE_SPLIT).map((c) => c.trim()).filter(Boolean);
    if (chunks.length < 2) return [];
    const out: Pair[] = [];
    for (const chunk of chunks) {
      const sub = this.splitOnce(chunk);
      if (sub.length) {
        for (const p of sub) {
          out.push({ ...p, separator: 'narrative', confidence: p.confidence * 0.85 });
        }
        continue;
      }
      const words = chunk.split(/\s+/).filter(Boolean);
      if (words.length < 2) continue;
      for (let cut = 1; cut < Math.min(words.length, 6); cut++) {
        out.push({
          labelText: words.slice(0, cut).join(' '),
          valueText: words.slice(cut).join(' '),
          separator: 'narrative-guess',
          confidence: 0.55 - 0.05 * cut,
        });
      }
    }
    return out;
  }

  buildCandidates(lines: OcrLine[]): LineCandidate[] {
    let current = this.defaultSection.id;
    const out: LineCandidate[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const text = line.text.trim();
      const [sectionId, isHeading] = this.detectSection(text, current);
      if (isHeading) current = sectionId;

      const prevText = i > 0 ? lines[i - 1].text : '';
      const nextText = i + 1 < lines.length ? lines[i + 1].text : '';

      let pairs: Pair[] = [];
      if (!isHeading && text) {
        pairs = this.splitOnce(text);
        if (!pairs.length || !pairs[0].valueText) {
          pairs = pairs.concat(this.splitByWords(text));
        }
        pairs = pairs.concat(this.splitNarrative(text));
        pairs.push({ labelText: prevText.trim(), valueText: text,
          separator: 'above', confidence: 0.45 });
        if (nextText) {
          pairs.push({ labelText: text, valueText: nextText.trim(),
            separator: 'below', confidence: 0.4 });
        }
        pairs = dedupe(pairs);
      }
      out.push({ line, sectionId: current, isHeading, pairs, prevText, nextText });
    }
    return out;
  }

  // ------------------------------------------------------------ matching
  labelSimilarity(text: string, field: FieldSpec): number {
    const norm = normaliseLabel(text);
    if (!norm) return 0;
    const normNoSpace = despace(norm);
    let best = 0;
    for (const form of field.labelForms) {
      if (!form) continue;
      if (norm === form) return 1;
      let score = Math.max(
        ratio(norm, form),
        tokenSetRatio(norm, form),
        ratio(normNoSpace, despace(form)),
      ) / 100;
      if (norm.includes(form) && norm !== form) {
        score = Math.max(score, 0.9 - 0.02 * (norm.length - form.length));
      }
      if (form.length <= 2) score *= norm.length > 3 ? 0.6 : 1;
      best = Math.max(best, score);
    }
    return Math.min(1, best);
  }

  private sectionBonus(field: FieldSpec, line: LineCandidate): number {
    const b = this.spec.matching.sectionBonus;
    if (line.sectionId === '__other__') return b.suppressed;
    if (field.section === line.sectionId) return b.sameSection;
    if (field.section === this.defaultSection.id) return b.defaultSection;
    return b.otherSection;
  }

  matchLine(line: LineCandidate, topK = 6): FieldMatch[] {
    if (line.isHeading) return [];
    const all: FieldMatch[] = [];
    for (const pair of line.pairs) {
      all.push(...this.matchPair(pair, line));
    }
    if (!all.length) return [];

    const bestPerField = new Map<string, FieldMatch>();
    for (const m of all) {
      const prev = bestPerField.get(m.key);
      if (!prev || m.score > prev.score) bestPerField.set(m.key, m);
    }
    return [...bestPerField.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private matchPair(pair: Pair, line: LineCandidate): FieldMatch[] {
    if (this.parser.isBlank(pair.valueText)) return [];
    const floor = this.spec.matching.labelScoreFloor;
    const weights = this.spec.matching.weights;

    const scored: Array<{
      field: FieldSpec; labelScore: number; parse: ParseResult; bonus: number;
    }> = [];
    for (const field of this.spec.fields) {
      const labelScore = this.labelSimilarity(pair.labelText, field);
      if (labelScore < floor) continue;
      scored.push({
        field, labelScore,
        parse: this.parser.parse(pair.valueText, field),
        bonus: this.sectionBonus(field, line),
      });
    }
    if (!scored.length) return [];

    const labelScores = scored.map((s) => s.labelScore).sort((a, b) => b - a);
    return scored.map(({ field, labelScore, parse, bonus }) => {
      const next = labelScores.find((s) => s < labelScore) ?? 0;
      const parseTerm = parse.valid ? parse.parseScore : 0;
      let score =
        weights.label * labelScore + weights.parse * parseTerm +
        weights.split * pair.confidence + bonus;
      if (!parse.valid) score *= this.spec.matching.invalidParsePenalty;
      score = Math.max(0, Math.min(1, score));
      return {
        key: field.key, field, pair, labelScore,
        labelMargin: labelScore - next, parse, sectionBonus: bonus,
        score, linkerScore: score, line,
      };
    }).sort((a, b) => b.score - a.score);
  }
}

function dedupe(pairs: Pair[]): Pair[] {
  const seen = new Map<string, Pair>();
  for (const p of pairs) {
    if (!p.labelText && !p.valueText) continue;
    const key = `${normaliseLabel(p.labelText)}\u0000${normaliseLabel(p.valueText)}`;
    const prev = seen.get(key);
    if (!prev || prev.confidence < p.confidence) seen.set(key, p);
  }
  return [...seen.values()].sort((a, b) => b.confidence - a.confidence);
}
