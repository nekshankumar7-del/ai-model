/**
 * Per-kind value parsers - the browser twin of src/factfind/parsers.py.
 *
 * A parser either returns a canonical, validated value or it refuses.
 * Refusing becomes "unsure" downstream; guessing becomes a wrong fill, and
 * the whole system is tuned to treat those as far worse. Every threshold and
 * pattern here comes from factfind-spec.json, so the Python side stays the
 * single source of truth.
 */
import { FieldSpec, ParseResult, FactFindSpec } from './spec.types';
import {
  extract, extractOne, normaliseLabel, ratio, titlecaseName, wRatio,
} from './text';

const DIGIT_CONFUSIONS: Record<string, string> = {
  O: '0', o: '0', D: '0', Q: '0',
  I: '1', l: '1', i: '1', '|': '1', '!': '1',
  Z: '2', z: '2', S: '5', s: '5', B: '8',
  G: '6', b: '6', T: '7', g: '9', q: '9',
};
const LETTER_CONFUSIONS: Record<string, string> = {
  '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A',
};

function translate(text: string, table: Record<string, string>): string {
  let out = '';
  for (const ch of text) out += table[ch] ?? ch;
  return out;
}

export function cleanRaw(text: unknown): string {
  if (text === null || text === undefined) return '';
  return String(text)
    .replace(/ /g, ' ')
    .replace(/^\s*(?:[-:•*>•–—]+\s*)+/u, '')
    .replace(/[\s.,;:\-_]+$/u, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function result(partial: Partial<ParseResult>): ParseResult {
  return {
    value: null, parseScore: 0, valid: false, candidates: [], reason: '', raw: '',
    ...partial,
  };
}

export class ValueParser {
  private readonly blank: Set<string>;
  private readonly yes: Set<string>;
  private readonly no: Set<string>;
  private readonly niPattern: RegExp;
  private readonly emailPattern: RegExp;

  constructor(private readonly spec: FactFindSpec) {
    this.blank = new Set(spec.parsing.blankTokens);
    this.yes = new Set(spec.parsing.yesForms);
    this.no = new Set(spec.parsing.noForms);
    this.niPattern = new RegExp(spec.parsing.niNumberPattern);
    this.emailPattern = new RegExp(spec.parsing.emailPattern);
  }

  isBlank(text: string): boolean {
    return this.blank.has(normaliseLabel(text)) || !cleanRaw(text);
  }

  parse(text: string, field: FieldSpec): ParseResult {
    const raw = cleanRaw(text);
    let out: ParseResult;
    switch (field.kind) {
      case 'yesNo': out = this.parseYesNo(raw); break;
      case 'choice': out = this.parseChoice(raw, field); break;
      case 'date': out = this.parseDate(raw, field); break;
      case 'number': out = this.parseNumber(raw, field); break;
      case 'money': out = this.parseMoney(raw); break;
      case 'email': out = this.parseEmail(raw); break;
      case 'phone': out = this.parsePhone(raw, field); break;
      case 'niNumber': out = this.parseNi(raw); break;
      case 'name':
      case 'fullName': out = this.parseName(raw, field); break;
      default: out = this.parseText(raw, field);
    }
    out.raw = out.raw || raw;
    return out;
  }

  // ---------------------------------------------------------------- yes/no
  private parseYesNo(raw: string): ParseResult {
    const norm = normaliseLabel(raw);
    if (this.isBlank(raw) && !this.yes.has(norm) && !this.no.has(norm)) {
      return result({ raw, reason: 'blank' });
    }
    const ambiguous = new Set(['x', 'tick', 'ticked']);

    if (this.no.has(norm)) {
      return result({ value: false, parseScore: 1, valid: true, raw,
        candidates: [[false, 1], [true, 0]] });
    }
    if (this.yes.has(norm)) {
      const score = ambiguous.has(norm) ? 0.55 : 1;
      return result({ value: true, parseScore: score, valid: true, raw,
        candidates: [[true, score], [false, 1 - score]] });
    }

    const first = norm.split(' ')[0] ?? '';
    if (this.no.has(first)) {
      return result({ value: false, parseScore: 0.8, valid: true, raw,
        candidates: [[false, 0.8], [true, 0.2]] });
    }
    if (this.yes.has(first)) {
      return result({ value: true, parseScore: 0.8, valid: true, raw,
        candidates: [[true, 0.8], [false, 0.2]] });
    }

    let bestYes = 0;
    let bestNo = 0;
    for (const f of this.yes) if (f.length > 2) bestYes = Math.max(bestYes, ratio(norm, f));
    for (const f of this.no) if (f.length > 2) bestNo = Math.max(bestNo, ratio(norm, f));
    if (Math.max(bestYes, bestNo) >= 85) {
      const value = bestYes >= bestNo;
      const score = (Math.max(bestYes, bestNo) / 100) * 0.8;
      return result({ value, parseScore: score, valid: true, raw,
        candidates: [[value, score], [!value, 1 - score]] });
    }
    return result({ raw, reason: `not a yes/no value: ${raw}`,
      candidates: [[true, 0], [false, 0]] });
  }

  // ---------------------------------------------------------------- choice
  private parseChoice(raw: string, field: FieldSpec): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    const norm = normaliseLabel(raw);
    const choices = field.choices ?? [];
    if (!choices.length) return result({ raw, reason: 'choice field with no choices' });

    const surfaceToValue = new Map<string, string>();
    for (const choice of choices) {
      for (const form of choice.surfaceForms) {
        if (!surfaceToValue.has(form)) surfaceToValue.set(form, choice.value);
      }
    }

    if (surfaceToValue.has(norm)) {
      const value = surfaceToValue.get(norm)!;
      const others = [...new Set(surfaceToValue.values())]
        .filter((v) => v !== value).slice(0, 2)
        .map((v) => [v, 0] as [unknown, number]);
      return result({ value, parseScore: 1, valid: true, raw,
        candidates: [[value, 1], ...others] });
    }

    const forms = [...surfaceToValue.keys()];
    const matches = extract(norm, forms, 12, wRatio);
    const ranked: Array<[string, number]> = [];
    const seen = new Set<string>();
    for (const [form, score] of matches) {
      const value = surfaceToValue.get(form)!;
      if (seen.has(value)) continue;
      seen.add(value);
      // Very short surface forms ("m", "s", "d") fuzzy-match almost anything.
      const penalty = form.length <= 2 && norm.length > 3 ? 0.75 : 1;
      ranked.push([value, (score / 100) * penalty]);
    }
    ranked.sort((a, b) => b[1] - a[1]);

    const snap = this.spec.parsing.choiceSnapThreshold;
    const marginFloor = this.spec.parsing.choiceMarginThreshold;
    if (ranked.length && ranked[0][1] >= snap) {
      const margin = ranked[0][1] - (ranked[1]?.[1] ?? 0);
      const score = ranked[0][1] * (margin >= marginFloor ? 1 : 0.7);
      return result({ value: ranked[0][0], parseScore: score, valid: true, raw,
        candidates: ranked.slice(0, 5) });
    }

    if (field.allowOther) {
      return result({ value: raw, parseScore: 0.35, valid: true, raw,
        reason: 'no choice matched; kept as other',
        candidates: [[raw, 0.35], ...ranked.slice(0, 4)] });
    }
    return result({ raw, reason: `no allowed choice matches ${raw}`,
      candidates: ranked.slice(0, 5) });
  }

  // ------------------------------------------------------------------ date
  private parseDate(raw: string, field: FieldSpec): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });

    let norm = raw.toLowerCase().replace(/,/g, ' ').replace(/\s+/g, ' ').trim();
    if (/^[\dOolISZB/\-.\s]+$/.test(norm)) {
      norm = translate(norm, DIGIT_CONFUSIONS).toLowerCase().replace(/\s+/g, ' ').trim();
    }

    const today = new Date();
    const patterns: Array<[RegExp, 'dmy' | 'ymd' | 'dMy' | 'Mdy']> = [
      [/^(\d{1,2})[/\-.\s](\d{1,2})[/\-.\s](\d{2}|\d{4})$/, 'dmy'],
      [/^(\d{4})[/\-.\s](\d{1,2})[/\-.\s](\d{1,2})$/, 'ymd'],
      [/^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?\s+(\d{2}|\d{4})$/, 'dMy'],
      [/^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2}|\d{4})$/, 'Mdy'],
      [/^(\d{2})(\d{2})(\d{2}|\d{4})$/, 'dmy'],
    ];

    for (const [pattern, order] of patterns) {
      const m = pattern.exec(norm);
      if (!m) continue;
      let day = 0;
      let month = 0;
      let year = 0;
      if (order === 'dmy') {
        day = +m[1]; month = +m[2]; year = expandYear(+m[3], today);
      } else if (order === 'ymd') {
        year = +m[1]; month = +m[2]; day = +m[3];
      } else if (order === 'dMy') {
        day = +m[1]; month = monthFromName(m[2]); year = expandYear(+m[3], today);
      } else {
        month = monthFromName(m[1]); day = +m[2]; year = expandYear(+m[3], today);
      }
      if (!month) continue;

      let iso = safeDate(year, month, day);
      if (!iso) {
        const swapped = safeDate(year, day, month);
        if (!swapped) return result({ raw, reason: `not a real date: ${raw}` });
        [day, month] = [month, day];
        iso = swapped;
      }

      let score = 1;
      let candidates: Array<[unknown, number]> = [[iso, 1]];
      if (order === 'dmy' && day <= 12 && month <= 12 && day !== month) {
        const alt = safeDate(year, day, month);
        if (alt) {
          // UK convention wins on a UK fact find, but not blindly.
          score = 0.72;
          candidates = [[iso, score], [alt, 0.28]];
        }
      }
      return this.applyDateRules(iso, candidates, score, field, raw, today);
    }
    return result({ raw, reason: `unparseable date: ${raw}` });
  }

  private applyDateRules(
    iso: string, candidates: Array<[unknown, number]>, score: number,
    field: FieldSpec, raw: string, today: Date,
  ): ParseResult {
    const [y, m, d] = iso.split('-').map(Number);
    if (y < 1900) return result({ raw, candidates, reason: `year ${y} before 1900` });

    if (field.minAge !== undefined) {
      const age = ageFrom(y, m, d, today);
      if (age < field.minAge) {
        return result({ raw, candidates,
          reason: `age ${age} below minAge ${field.minAge}` });
      }
      if (age > 120) return result({ raw, candidates, reason: `implausible age ${age}` });
    }
    const isBirth = field.field.toLowerCase().includes('birth') || field.minAge !== undefined;
    if (isBirth && new Date(iso) > today) {
      return result({ raw, candidates, reason: 'birth date in the future' });
    }
    return result({ value: iso, parseScore: score, valid: true, raw, candidates });
  }

  // ------------------------------------------------------------- number(s)
  private extractNumber(raw: string): { value: number | null; score: number; note: string } {
    let text = raw.trim().replace(/[£$€]/g, '').replace(/ /g, ' ');
    let mult = 1;
    let note = '';
    const m = /(\d)\s*(k|m|bn)\b/i.exec(text);
    if (m) {
      mult = { k: 1000, m: 1000000, bn: 1000000000 }[m[2].toLowerCase()]!;
      text = text.replace(/\s*(k|m|bn)\b/i, '');
      note = `x${mult}`;
    }

    const core = /-?[\d.,\s]+/.exec(text);
    if (!core) return { value: null, score: 0, note: 'no digits' };
    let s = core[0].trim().replace(/(?<=\d)\s+(?=\d)/g, '');
    if (!/\d/.test(s)) return { value: null, score: 0, note: 'no digits' };

    let score = 1;
    const hasComma = s.includes(',');
    const hasDot = s.includes('.');
    if (hasComma && hasDot) {
      s = s.lastIndexOf(',') > s.lastIndexOf('.')
        ? s.replace(/\./g, '').replace(',', '.')
        : s.replace(/,/g, '');
    } else if (hasComma) {
      const parts = s.split(',');
      const last = parts[parts.length - 1];
      if (last.length === 3 && parts.slice(0, -1).every((p) => p.length <= 3)) {
        s = s.replace(/,/g, '');
      } else if (last.length === 1 || last.length === 2) {
        s = s.replace(',', '.');
      } else {
        s = s.replace(/,/g, '');
        score = 0.8;
      }
    } else if (hasDot) {
      const parts = s.split('.');
      if (parts.length > 2) {
        s = s.replace(/\./g, '');
      } else if (parts[parts.length - 1].length === 3 && parts[0].length <= 3) {
        // 1.200 is ambiguous: 1200 or 1.2? Thousands is far more common here.
        s = s.replace(/\./g, '');
        score = 0.7;
        note = `${note} ambiguous-thousands`.trim();
      }
    }
    const value = Number(s) * mult;
    if (!Number.isFinite(value)) {
      return { value: null, score: 0, note: 'unparseable number' };
    }
    return { value, score, note };
  }

  private parseNumber(raw: string, field: FieldSpec): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    let { value, score, note } = this.extractNumber(raw);
    if (value === null) return result({ raw, reason: note });

    if (!Number.isInteger(value)) {
      if (Math.abs(value - Math.round(value)) < 1e-9) value = Math.round(value);
      else score *= 0.6;
    }
    const out = Number.isInteger(value) ? Math.round(value) : value;

    if (field.min !== undefined && out < field.min) {
      return result({ raw, candidates: [[out, score]],
        reason: `${out} below min ${field.min}` });
    }
    if (field.max !== undefined && out > field.max) {
      return result({ raw, candidates: [[out, score]],
        reason: `${out} above max ${field.max}` });
    }
    if (note) score *= 0.9;
    return result({ value: out, parseScore: score, valid: true, raw,
      candidates: [[out, score]], reason: note });
  }

  private parseMoney(raw: string): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    let { value, score, note } = this.extractNumber(raw);
    if (value === null) return result({ raw, reason: note });
    if (value < 0) return result({ raw, reason: 'negative amount' });
    if (value > 100000000) {
      return result({ raw, candidates: [[value, score]],
        reason: `implausible amount ${value}` });
    }

    const period = /\b(p\s*\/?\s*m|per\s+month|pm|monthly|a\s+month|p\s*\/?\s*w|per\s+week|pw|weekly|a\s+week|p\s*\/?\s*a|per\s+annum|pa|per\s+year|annually|yearly|a\s+year|per\s+quarter|quarterly)\b/i
      .exec(raw);
    let candidates: Array<[unknown, number]> = [];
    const out = Math.round(value * 100) / 100;
    if (period) {
      // Monthly vs annual salary is a 12x silent error: do not fill it blind.
      note = `${note} period:${period[0].trim()}`.trim();
      score *= 0.5;
      candidates = [[out, score]];
      if (/month|pm|p\/m/i.test(period[0])) {
        candidates.push([Math.round(value * 12 * 100) / 100, score * 0.8]);
      }
    }
    if (!candidates.length) candidates = [[out, score]];
    return result({ value: out, parseScore: score, valid: true, raw,
      candidates, reason: note });
  }

  // ------------------------------------------------------- identity values
  private parseEmail(raw: string): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    let s = raw.replace(/\s/g, '').trim().replace(/^[.,;:]+|[.,;:]+$/g, '');
    s = s.replace(/\s*\(?at\)?\s*@/gi, '@')
      .replace(/^(e\s*-?\s*mail|email)\s*:?/i, '')
      .replace(/[’']/g, '');
    if (!s.includes('@')) {
      const m = /(.+?)\s*\(?\s*at\s*\)?\s*(.+\..+)/i.exec(raw);
      if (m) s = `${m[1].trim()}@${m[2].trim()}`.replace(/\s/g, '');
    }
    s = s.toLowerCase();

    const commonDomains = [
      'gmail.com', 'googlemail.com', 'hotmail.com', 'hotmail.co.uk',
      'outlook.com', 'yahoo.com', 'yahoo.co.uk', 'icloud.com', 'live.co.uk',
      'btinternet.com', 'sky.com', 'aol.com', 'msn.com', 'me.com',
      'protonmail.com', 'virginmedia.com',
    ];

    if (this.emailPattern.test(s)) {
      const [local, domain] = s.split('@');
      let score = 1;
      const candidates: Array<[unknown, number]> = [[s, 1]];
      if (!commonDomains.includes(domain)) {
        const near = extractOne(domain, commonDomains, ratio);
        if (near && near[1] >= 80 && near[1] < 100) {
          // Plausible OCR slip. Offer the fix; never rewrite it silently.
          score = 0.6;
          candidates.push([`${local}@${near[0]}`, 0.55]);
        }
      }
      return result({ value: s, parseScore: score, valid: true, raw, candidates });
    }

    const repaired = s.replace(/\.{2,}/g, '.').replace(/@+/g, '@');
    if (this.emailPattern.test(repaired)) {
      return result({ raw, candidates: [[repaired, 0.5]],
        reason: 'email only valid after repair; left unsure' });
    }
    return result({ raw, reason: `not a valid email: ${raw}` });
  }

  private parsePhone(raw: string, field: FieldSpec): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    const hadPlus = raw.trimStart().startsWith('+');
    let d = translate(raw, DIGIT_CONFUSIONS).replace(/\D/g, '');
    if (!d) return result({ raw, reason: 'no digits' });

    if (d.startsWith('0044')) d = '0' + d.slice(4);
    else if (hadPlus && d.startsWith('44')) d = '0' + d.slice(2);
    else if (d.startsWith('44') && d.length === 12) d = '0' + d.slice(2);

    let penalty = 1;
    if (d.length === 10 && '1237'.includes(d[0]) && !d.startsWith('0')) {
      d = '0' + d;                       // leading zero dropped by the writer
      penalty = 0.85;
    }
    if (d.length !== this.spec.parsing.ukPhone.digits || !d.startsWith('0')) {
      return result({ raw, candidates: [[d, 0.3]],
        reason: `UK number must be 11 digits starting 0, got ${d.length}` });
    }
    if (d.startsWith('00')) return result({ raw, reason: 'invalid prefix 00' });

    let score = penalty;
    const wantsMobile = /mobile/i.test(field.field) || /mobile/i.test(field.label);
    if (wantsMobile && !d.startsWith(this.spec.parsing.ukPhone.mobilePrefix)) {
      score *= 0.6;
    }
    if (raw.replace(/\D/g, '').length < translate(raw, DIGIT_CONFUSIONS).replace(/\D/g, '').length) {
      score *= 0.85;                     // heavy digit repair was needed
    }
    return result({ value: d, parseScore: score, valid: true, raw,
      candidates: [[d, score]] });
  }

  private parseNi(raw: string): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    let s = raw.replace(/[\s\-.]/g, '').toUpperCase();
    s = s.replace(/^(NI|NINO|NINUMBER|NATIONALINSURANCE(NUMBER)?):?/, '');

    if (this.niPattern.test(s)) {
      return result({ value: s, parseScore: 1, valid: true, raw,
        candidates: [[s, 1]] });
    }
    if (s.length === 9) {
      const fixed =
        translate(s.slice(0, 2), LETTER_CONFUSIONS) +
        translate(s.slice(2, 8), DIGIT_CONFUSIONS) +
        translate(s.slice(8), LETTER_CONFUSIONS);
      if (this.niPattern.test(fixed)) {
        return result({ raw, candidates: [[fixed, 0.5]],
          reason: 'NI number only valid after character repair; left unsure' });
      }
    }
    return result({ raw, reason: `does not match NI number pattern: ${raw}` });
  }

  // ------------------------------------------------------- names and text
  private parseName(raw: string, field: FieldSpec): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    const cleaned = raw.replace(/[^A-Za-z'’\-.\s]/g, ' ')
      .replace(/\s+/g, ' ').replace(/^[\s.]+|[\s.]+$/g, '');
    if (!cleaned) return result({ raw, reason: 'no letters in name' });
    if (!/^[A-Za-z][A-Za-z'’\-.\s]{0,78}$/.test(cleaned)) {
      return result({ raw, reason: `not name-like: ${raw}` });
    }

    const titles = new Set(['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'lady', 'rev']);
    let parts = cleaned.split(/\s+/).filter(Boolean);
    if (parts.length && titles.has(normaliseLabel(parts[0]).replace(/\.$/, ''))) {
      parts = parts.slice(1);
    }
    if (!parts.length) return result({ raw, reason: 'only a title, no name' });

    let score = 1;
    if (raw.length !== cleaned.length) score *= 0.9;
    if (parts.some((t) => t.length === 1 && !t.endsWith('.'))) score *= 0.9;

    if (field.kind === 'fullName' && parts.length < 2) score *= 0.55;
    if (field.kind === 'name' && parts.length > 1
        && ['firstName', 'lastName', 'middleName'].includes(field.field)) {
      score *= 0.8;
    }

    const value = titlecaseName(parts.join(' '));
    if (value.length < 2) return result({ raw, reason: 'name too short' });
    return result({ value, parseScore: score, valid: true, raw,
      candidates: [[value, score]] });
  }

  private parseText(raw: string, field: FieldSpec): ParseResult {
    if (this.isBlank(raw)) return result({ raw, reason: 'blank' });
    let s = raw.replace(/\s+/g, ' ').trim();
    if (s.length > 500) s = s.slice(0, 500);

    let score = 0.85;                    // free text is never fully certain
    let candidates: Array<[unknown, number]> = [];

    if (field.vocabulary?.length) {
      const norm = normaliseLabel(s);
      const normalised = field.vocabulary.map(normaliseLabel);
      const matches = extract(norm, normalised, 5, wRatio);
      const ranked = matches.map(([, sc, i]) =>
        [field.vocabulary![i], sc / 100] as [string, number]);
      if (ranked.length) {
        const [best, bestScore] = ranked[0];
        if (bestScore >= this.spec.parsing.vocabularySnapThreshold) {
          return result({ value: best, parseScore: Math.min(1, bestScore),
            valid: true, raw, candidates: ranked.slice(0, 5) });
        }
        candidates = ranked.slice(0, 5);
        if (bestScore >= 0.8) score = 0.7;
      }
    }

    candidates = candidates.length
      ? [[s, score], ...candidates.slice(0, 4)]
      : [[s, score]];
    return result({ value: s, parseScore: score, valid: true, raw, candidates });
  }
}

// ---------------------------------------------------------------- helpers

function expandYear(y: number, today: Date): number {
  if (y >= 100) return y;
  const century = Math.floor(today.getFullYear() / 100) * 100;
  const candidate = century + y;
  return candidate <= today.getFullYear() + 10 ? candidate : candidate - 100;
}

const MONTH_NAMES = [
  'january', 'february', 'march', 'april', 'may', 'june',
  'july', 'august', 'september', 'october', 'november', 'december',
];

function monthFromName(name: string): number {
  const n = name.toLowerCase().replace(/\.$/, '');
  const extra: Record<string, number> = { sept: 9, septmber: 9, febuary: 2 };
  if (extra[n]) return extra[n];
  for (let i = 0; i < 12; i++) {
    if (MONTH_NAMES[i] === n || MONTH_NAMES[i].slice(0, 3) === n) return i + 1;
  }
  return 0;
}

function safeDate(y: number, m: number, d: number): string | null {
  if (m < 1 || m > 12 || d < 1 || d > 31) return null;
  const date = new Date(Date.UTC(y, m - 1, d));
  if (date.getUTCFullYear() !== y || date.getUTCMonth() !== m - 1
      || date.getUTCDate() !== d) {
    return null;
  }
  return `${String(y).padStart(4, '0')}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

function ageFrom(y: number, m: number, d: number, today: Date): number {
  let age = today.getFullYear() - y;
  const beforeBirthday =
    today.getMonth() + 1 < m || (today.getMonth() + 1 === m && today.getDate() < d);
  if (beforeBirthday) age -= 1;
  return age;
}
