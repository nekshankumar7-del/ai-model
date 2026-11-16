/**
 * FactFindExtractionService - runs the whole extraction pipeline in the
 * browser. No server, no external AI API: the ONNX models run under
 * onnxruntime-web and the document never leaves the page.
 *
 *   const result = await service.extractFiles(fileList);
 *   patchForm(result.sections);         // only the 'filled' values
 *   showForReview(result.fields.filter(f => f.status === 'unsure'));
 *
 * The decision rule is the point of the whole thing: a value is only filled
 * when the calibrated probability that it is correct clears the threshold for
 * that field's kind. Everything else comes back as 'unsure' with its top
 * three candidates, so the adviser chooses instead of the model guessing.
 */
import { Injectable } from '@angular/core';
import type { InferenceSession, Tensor } from 'onnxruntime-web';

import {
  ExtractionResult, FactFindSpec, FieldPrediction, FieldSpec, OcrLine,
  OcrReader, PageQuality,
} from './spec.types';
import { FeatureBuilder, applyCalibration, assertFeatureOrder } from './features';
import { FieldMatch, LineCandidate, Matcher } from './matcher';
import { PagePreprocessor, PreparedPage } from './preprocess';

export interface ExtractionOptions {
  /** Shifts every threshold. Positive fills less, negative fills more. */
  thresholdShift?: number;
  /** Only consider fields for this case type ('mortgage' | 'protection'). */
  caseType?: string;
  onProgress?: (done: number, total: number, label: string) => void;
}

const ORT_IMPORT = () => import('onnxruntime-web');

@Injectable({ providedIn: 'root' })
export class FactFindExtractionService {
  private spec?: FactFindSpec;
  private matcher?: Matcher;
  private featureBuilder?: FeatureBuilder;
  private preprocessor = new PagePreprocessor();
  private linkerSession?: InferenceSession;
  private confidenceSession?: InferenceSession;
  private ready?: Promise<void>;

  /** Where the exported assets live, relative to the app's base href. */
  assetBase = 'assets/factfind';

  /** Wired in by the host app; see ocr/README for the two supported readers. */
  ocr?: OcrReader;

  // ------------------------------------------------------------- loading
  init(): Promise<void> {
    if (!this.ready) this.ready = this.load();
    return this.ready;
  }

  private async load(): Promise<void> {
    const response = await fetch(`${this.assetBase}/factfind-spec.json`);
    if (!response.ok) {
      throw new Error(
        `could not load ${this.assetBase}/factfind-spec.json (${response.status}). ` +
        `Run scripts/05_export_onnx.py and copy angular/assets/factfind into ` +
        `your app's assets folder.`,
      );
    }
    const spec = (await response.json()) as FactFindSpec;
    this.spec = spec;
    this.matcher = new Matcher(spec);
    this.featureBuilder = new FeatureBuilder(spec);

    const ort = await ORT_IMPORT();
    const options: InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    };

    if (spec.models.linker.kept) {
      this.linkerSession = await ort.InferenceSession.create(
        `${this.assetBase}/${spec.models.linker.file}`, options,
      );
      assertFeatureOrder(
        spec.models.linker.featureNames,
        inputWidth(this.linkerSession, spec.models.linker.inputName),
      );
    }
    if (spec.models.confidence.kept) {
      this.confidenceSession = await ort.InferenceSession.create(
        `${this.assetBase}/${spec.models.confidence.file}`, options,
      );
      assertFeatureOrder(
        spec.models.confidence.featureNames,
        inputWidth(this.confidenceSession, spec.models.confidence.inputName),
      );
    }
  }

  // ------------------------------------------------------------ entry points
  /** Extract from several files and merge them into one fact find. */
  async extractFiles(
    files: FileList | File[],
    options: ExtractionOptions = {},
  ): Promise<{
    perFile: ExtractionResult[];
    merged: Record<string, unknown>;
    conflicts: Array<Record<string, unknown>>;
  }> {
    const list = Array.from(files as ArrayLike<File>);
    const perFile: ExtractionResult[] = [];
    for (let i = 0; i < list.length; i++) {
      options.onProgress?.(i, list.length, list[i].name);
      perFile.push(await this.extractFile(list[i], options));
    }
    options.onProgress?.(list.length, list.length, 'done');
    return { perFile, ...mergeResults(perFile) };
  }

  async extractFile(file: File, options: ExtractionOptions = {}): Promise<ExtractionResult> {
    await this.init();
    const pages = await this.preprocessor.loadFile(file);
    return this.extractPages(pages, file.name, options);
  }

  async extractPages(
    pages: PreparedPage[],
    source: string,
    options: ExtractionOptions = {},
  ): Promise<ExtractionResult> {
    await this.init();
    const spec = this.spec!;
    const matcher = this.matcher!;

    if (!this.ocr) {
      throw new Error(
        'no OCR reader is wired in. Set `service.ocr` to a TesseractOcrReader ' +
        'or an OnnxOcrReader before extracting; see angular/src/factfind/ocr/.',
      );
    }

    const allMatches: FieldMatch[] = [];
    const qualityByPage = new Map<number, PageQuality>();
    let lineCount = 0;

    for (let i = 0; i < pages.length; i++) {
      const page = pages[i];
      qualityByPage.set(i, page.quality);
      const lines = await this.ocr.read(page.canvas, i);
      lineCount += lines.length;
      const candidates = matcher.buildCandidates(lines);
      for (const candidate of candidates) {
        let top = matcher.matchLine(candidate, 6);
        if (top.length) top = await this.rescore(top);
        allMatches.push(...top);
      }
    }

    const chosen = this.assign(allMatches, options.caseType);
    const flags = crossFieldFlags(chosen, spec);
    const fields = await this.decide(chosen, flags, qualityByPage, options);

    const warnings: string[] = [];
    if (!lineCount) warnings.push('no text was read from this document');

    return {
      source,
      sections: toSections(fields.filter((f) => f.status === 'filled')),
      fields,
      summary: {
        filled: fields.filter((f) => f.status === 'filled').length,
        unsure: fields.filter((f) => f.status === 'unsure').length,
        pages: pages.length,
      },
      warnings,
    };
  }

  // ----------------------------------------------------------- the models
  private async rescore(matches: FieldMatch[]): Promise<FieldMatch[]> {
    const spec = this.spec!;
    if (!this.linkerSession || !spec.models.linker.kept) return matches;

    const names = spec.models.linker.featureNames;
    const builder = this.featureBuilder!;
    const defaultSection = this.matcher!.defaultSectionId;

    const batch = new Float32Array(matches.length * names.length);
    matches.forEach((match, row) => {
      const vector = builder.vector(builder.linkerFeatures(match, defaultSection), names);
      batch.set(vector, row * names.length);
    });

    const probabilities = await this.run(
      this.linkerSession, spec.models.linker.inputName, batch,
      matches.length, names.length, spec.models.linker.positiveClassIndex,
    );

    const blend = spec.models.linker.blend ?? 0.65;
    matches.forEach((match, i) => {
      match.linkerScore = probabilities[i];
      match.score = blend * probabilities[i] + (1 - blend) * match.score;
    });
    return matches.sort((a, b) => b.score - a.score);
  }

  private async decide(
    chosen: Map<string, FieldMatch>,
    flags: Map<string, Record<string, number>>,
    qualityByPage: Map<number, PageQuality>,
    options: ExtractionOptions,
  ): Promise<FieldPrediction[]> {
    const spec = this.spec!;
    const builder = this.featureBuilder!;
    const defaultSection = this.matcher!.defaultSectionId;
    const entries = [...chosen.entries()].sort((a, b) => a[0].localeCompare(b[0]));

    let probabilities: number[];
    if (this.confidenceSession && spec.models.confidence.kept && entries.length) {
      const names = spec.models.confidence.featureNames;
      const batch = new Float32Array(entries.length * names.length);
      entries.forEach(([key, match], row) => {
        const bag = builder.confidenceFeatures(
          match, defaultSection, match.linkerScore,
          flags.get(key) ?? {},
          qualityByPage.get(match.line.line.pageIndex) ?? ({} as PageQuality),
        );
        batch.set(builder.vector(bag, names), row * names.length);
      });
      const raw = await this.run(
        this.confidenceSession, spec.models.confidence.inputName, batch,
        entries.length, names.length, spec.models.confidence.positiveClassIndex,
      );
      probabilities = Array.from(raw, (p) =>
        applyCalibration(p, spec.models.confidence.calibration));
    } else {
      // No confidence model: the rule score stands in. It is not calibrated.
      probabilities = entries.map(([, m]) => m.score);
    }

    return entries.map(([key, match], i) => {
      const confidence = probabilities[i];
      const hardBlock = (flags.get(key)?.['only_if_violated'] ?? 0) > 0;
      const threshold = this.thresholdFor(match.field, options.thresholdShift ?? 0);
      const status: FieldPrediction['status'] =
        confidence >= threshold && match.parse.valid && !hardBlock ? 'filled' : 'unsure';

      let reason = match.parse.reason;
      if (hardBlock) reason = `${reason} | onlyIf condition not met`.replace(/^\s*\|\s*/, '');

      return {
        key,
        field: match.key,
        kind: match.field.kind,
        label: match.field.label,
        value: status === 'filled' ? match.parse.value : null,
        status,
        confidence: Math.round(confidence * 10000) / 10000,
        candidates: match.parse.candidates
          .slice(0, spec.decision.unsureTopK).map(([c]) => c),
        raw: match.parse.raw,
        page: match.line.line.pageIndex,
        box: match.line.line.box,
        reason,
      };
    });
  }

  private thresholdFor(field: FieldSpec, shift: number): number {
    const spec = this.spec!;
    const thresholds = spec.decision.thresholds;
    let base = thresholds[field.key] ?? thresholds[field.kind] ?? thresholds['default'] ?? 0.5;
    base += shift;
    const floor = spec.decision.hardFloors[field.kind];
    if (floor !== undefined) base = Math.max(base, floor);
    return Math.min(1.01, Math.max(0, base));
  }

  private async run(
    session: InferenceSession, inputName: string, batch: Float32Array,
    rows: number, cols: number, positiveIndex: number,
  ): Promise<Float32Array> {
    const ort = await ORT_IMPORT();
    const tensor = new ort.Tensor('float32', batch, [rows, cols]);
    const outputs = await session.run({ [inputName]: tensor });
    for (const value of Object.values(outputs) as Tensor[]) {
      if (value.dims?.length === 2 && value.dims[1] === 2) {
        const data = value.data as Float32Array;
        const out = new Float32Array(rows);
        for (let i = 0; i < rows; i++) out[i] = data[i * 2 + positiveIndex];
        return out;
      }
    }
    throw new Error('no 2-column probability output found in the ONNX result');
  }

  // --------------------------------------------------------- assignment
  private assign(matches: FieldMatch[], caseType?: string): Map<string, FieldMatch> {
    const spec = this.spec!;
    const minScore = spec.matching.assignMinScore;
    const repeating = new Set(spec.sections.filter((s) => s.repeating).map((s) => s.id));

    const chosen = new Map<string, FieldMatch>();
    const usedLines = new Set<string>();
    const rows = new Map<string, Array<{ keys: Set<string>; y: number; n: number }>>();

    for (const match of [...matches].sort((a, b) => b.score - a.score)) {
      if (match.score < minScore) continue;
      if (caseType && match.field.when && match.field.when !== caseType) continue;

      const lineId = `${match.line.line.pageIndex}:${match.line.line.lineIndex}`;
      if (usedLines.has(lineId)) continue;      // one line fills one field

      let flatKey = match.key;
      if (repeating.has(match.field.section)) {
        const index = rowIndexFor(match, rows);
        flatKey = `${match.field.section}[${index}].${match.field.field}`;
      }
      if (chosen.has(flatKey)) continue;
      chosen.set(flatKey, match);
      usedLines.add(lineId);
    }
    return chosen;
  }
}

// ------------------------------------------------------------- helpers

function inputWidth(session: InferenceSession, inputName: string): number {
  const meta = (session as unknown as {
    inputMetadata?: Record<string, { dimensions?: number[] }>;
  }).inputMetadata;
  const dims = meta?.[inputName]?.dimensions;
  // Fall back to trusting the spec when the runtime does not expose metadata.
  return dims && dims.length === 2 && dims[1] > 0 ? dims[1] : -1;
}

function rowIndexFor(
  match: FieldMatch,
  rows: Map<string, Array<{ keys: Set<string>; y: number; n: number }>>,
): number {
  const sectionId = match.field.section;
  const list = rows.get(sectionId) ?? [];
  if (!rows.has(sectionId)) rows.set(sectionId, list);

  const [, y, , h] = match.line.line.box;
  const centre = y + h / 2;
  const lineHeight = Math.max(1, h);

  for (let i = 0; i < list.length; i++) {
    const row = list[i];
    if (row.keys.has(match.field.field)) continue;
    if (Math.abs(centre - row.y) <= lineHeight * 9) {
      row.keys.add(match.field.field);
      row.y = (row.y * row.n + centre) / (row.n + 1);
      row.n += 1;
      return i;
    }
  }
  list.push({ keys: new Set([match.field.field]), y: centre, n: 1 });
  return list.length - 1;
}

/**
 * Cross-field consistency signals. These catch the errors a single-field view
 * cannot see - a dependant older than the client, a retirement age already
 * passed, an onlyIf condition contradicted.
 */
function crossFieldFlags(
  chosen: Map<string, FieldMatch>,
  spec: FactFindSpec,
): Map<string, Record<string, number>> {
  const out = new Map<string, Record<string, number>>();
  const values = new Map<string, unknown>();
  for (const [key, match] of chosen) values.set(key, match.parse.value);

  const today = new Date();
  const ageOf = (iso: unknown): number | null => {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso ?? ''));
    if (!m) return null;
    let age = today.getFullYear() - +m[1];
    const month = today.getMonth() + 1;
    if (month < +m[2] || (month === +m[2] && today.getDate() < +m[3])) age -= 1;
    return age;
  };
  const clientAge = ageOf(values.get('personal-details.dateOfBirth'));

  for (const [key, match] of chosen) {
    const flags: Record<string, number> = {};
    const field = match.field;

    if (field.field === 'retirementAge' && clientAge !== null) {
      const ra = Number(match.parse.value);
      if (Number.isFinite(ra)) {
        flags['retirement_before_age'] = ra < clientAge - 1 ? 1 : 0;
        flags['retirement_gap'] = ra - clientAge;
      }
    }

    if (field.section === 'dependants' && field.field === 'dateOfBirth'
        && clientAge !== null) {
      const depAge = ageOf(match.parse.value);
      if (depAge !== null) {
        flags['dependant_older_than_client'] = depAge > clientAge ? 1 : 0;
        flags['dependant_age'] = depAge;
        const rel = values.get(key.replace('.dateOfBirth', '.relationshipType'));
        if ((rel === 'Son' || rel === 'Daughter') && depAge >= clientAge) {
          flags['relationship_age_conflict'] = 1;
        }
      }
    }

    if (field.field === 'dependencyTermYears') {
      const term = Number(match.parse.value);
      if (Number.isFinite(term)) flags['term_implausible'] = term > 30 ? 1 : 0;
    }

    if (field.kind === 'name' || field.kind === 'fullName') {
      const surnames: string[] = [];
      for (const [otherKey, other] of chosen) {
        if (otherKey === key) continue;
        if (other.field.kind !== 'name' && other.field.kind !== 'fullName') continue;
        const parts = String(other.parse.value ?? '').split(/\s+/).filter(Boolean);
        if (parts.length) surnames.push(parts[parts.length - 1].toLowerCase());
      }
      const mine = String(match.parse.value ?? '').split(/\s+/).filter(Boolean);
      if (mine.length && surnames.length) {
        flags['surname_shared'] =
          surnames.includes(mine[mine.length - 1].toLowerCase()) ? 1 : 0;
      }
    }

    if (field.kind === 'money') {
      const others: number[] = [];
      for (const [otherKey, other] of chosen) {
        if (otherKey === key || other.field.kind !== 'money') continue;
        const v = Number(other.parse.value);
        if (Number.isFinite(v)) others.push(v);
      }
      const mine = Number(match.parse.value);
      if (others.length >= 3 && Number.isFinite(mine)) {
        others.sort((a, b) => a - b);
        const median = others[Math.floor(others.length / 2)];
        if (median > 0) flags['money_ratio_to_median'] = Math.min(100, mine / median);
      }
    }

    if (field.onlyIf) {
      const prefix = key.includes('[') ? key.slice(0, key.lastIndexOf('.')) : field.section;
      const depKey = `${prefix}.${field.onlyIf.key}`;
      const known = values.has(depKey);
      const depValue = values.get(depKey);
      const satisfied = field.onlyIf.filled !== undefined
        ? (depValue !== null && depValue !== undefined && depValue !== '') === field.onlyIf.filled
        : depValue === field.onlyIf.equals;
      flags['only_if_violated'] = known && !satisfied ? 1 : 0;
    }

    out.set(key, flags);
  }
  return out;
}

function toSections(fields: FieldPrediction[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const prediction of fields) {
    const dot = prediction.key.indexOf('.');
    const section = prediction.key.slice(0, dot);
    const rest = prediction.key.slice(dot + 1);
    if (section.endsWith(']')) {
      const open = section.lastIndexOf('[');
      const base = section.slice(0, open);
      const index = Number(section.slice(open + 1, -1));
      const rows = (out[base] as Array<Record<string, unknown>>) ?? [];
      out[base] = rows;
      while (rows.length <= index) rows.push({});
      rows[index][rest] = prediction.value;
    } else {
      const bucket = (out[section] as Record<string, unknown>) ?? {};
      out[section] = bucket;
      bucket[rest] = prediction.value;
    }
  }
  return out;
}

/** Merge several documents; the more confident value wins, conflicts recorded. */
function mergeResults(results: ExtractionResult[]): {
  merged: Record<string, unknown>;
  conflicts: Array<Record<string, unknown>>;
} {
  const best = new Map<string, FieldPrediction & { source: string }>();
  const conflicts: Array<Record<string, unknown>> = [];

  for (const result of results) {
    for (const field of result.fields) {
      if (field.status !== 'filled') continue;
      const current = best.get(field.key);
      if (!current) {
        best.set(field.key, { ...field, source: result.source });
        continue;
      }
      if (String(current.value) !== String(field.value)) {
        conflicts.push({
          field: field.key, kept: current.value, from: current.source,
          discarded: field.value, alsoIn: result.source,
          keptConfidence: current.confidence,
          discardedConfidence: field.confidence,
        });
      }
      if (field.confidence > current.confidence) {
        best.set(field.key, { ...field, source: result.source });
      }
    }
  }
  return { merged: toSections([...best.values()]), conflicts };
}
