/**
 * Feature vectors for the two ONNX models - the browser twin of features.py.
 *
 * Correctness here is not optional: the vector must match the training order
 * and scaling exactly, or the model's calibration is meaningless and it will
 * confidently fill wrong values. The order comes from the spec, never from
 * this file, and `assertFeatureOrder` checks the spec against what the model
 * expects at startup.
 */
import { FactFindSpec, PageQuality } from './spec.types';
import { FieldMatch } from './matcher';

export type FeatureBag = Record<string, number>;

export class FeatureBuilder {
  private readonly logScaled: Set<string>;
  private readonly div255: Set<string>;
  private readonly kindOrder: string[];
  private readonly separatorOrder: string[];

  constructor(private readonly spec: FactFindSpec) {
    this.logScaled = new Set(spec.features.scaling.log1pSigned);
    this.div255 = new Set(spec.features.scaling.divide255);
    this.kindOrder = spec.features.linker.kindOrder;
    this.separatorOrder = spec.features.linker.separatorOrder;
  }

  /** Base features shared by both models. */
  private base(match: FieldMatch): FeatureBag {
    const line = match.line.line;
    const pw = Math.max(1, line.pageWidth);
    const ph = Math.max(1, line.pageHeight);
    const [x, y, w, h] = line.box;
    const candidates = match.parse.candidates;
    const candidateMargin = candidates.length < 2
      ? (candidates.length ? 1 : 0)
      : candidates[0][1] - candidates[1][1];

    return {
      label_score: match.labelScore,
      label_margin: match.labelMargin,
      parse_score: match.parse.parseScore,
      parse_valid: match.parse.valid ? 1 : 0,
      split_confidence: match.pair.confidence,
      ocr_confidence: line.confidence,
      section_bonus: match.sectionBonus,
      combined_score: match.score,
      rel_x: x / pw, rel_y: y / ph, rel_w: w / pw, rel_h: h / ph,
      label_len: match.pair.labelText.length,
      value_len: match.pair.valueText.length,
      line_len: line.text.length,
      n_candidates: candidates.length,
      candidate_margin: candidateMargin,
      is_sensitive: ['niNumber', 'email', 'phone'].includes(match.field.kind) ? 1 : 0,
      value_has_digit: /\d/.test(match.pair.valueText) ? 1 : 0,
      value_has_alpha: /[A-Za-z]/.test(match.pair.valueText) ? 1 : 0,
    };
  }

  linkerFeatures(match: FieldMatch, defaultSectionId: string): FeatureBag {
    const bag = this.base(match);
    for (const kind of this.kindOrder) {
      bag[`kind_${kind}`] = match.field.kind === kind ? 1 : 0;
    }
    for (const sep of this.separatorOrder) {
      bag[`sep_${sep}`] = match.pair.separator === sep ? 1 : 0;
    }
    bag['same_section'] = match.field.section === match.line.sectionId ? 1 : 0;
    bag['is_default_section'] = match.field.section === defaultSectionId ? 1 : 0;
    bag['is_other_section'] = match.line.sectionId === '__other__' ? 1 : 0;
    return bag;
  }

  confidenceFeatures(
    match: FieldMatch,
    defaultSectionId: string,
    linkerScore: number,
    flags: Record<string, number>,
    image: Partial<PageQuality>,
  ): FeatureBag {
    const bag = this.linkerFeatures(match, defaultSectionId);
    bag['linker_score'] = linkerScore;
    for (const name of this.spec.features.confidence.featureNames) {
      if (name.startsWith('flag_')) {
        bag[name] = flags[name.slice(5)] ?? 0;
      } else if (name.startsWith('img_')) {
        const value = (image as Record<string, number | undefined>)[name.slice(4)];
        bag[name] = typeof value === 'number' ? value : 0;
      }
    }
    return bag;
  }

  /** Project a bag onto the fixed, scaled vector the model expects. */
  vector(bag: FeatureBag, names: string[]): Float32Array {
    const out = new Float32Array(names.length);
    for (let i = 0; i < names.length; i++) {
      out[i] = this.scale(names[i], bag[names[i]] ?? 0);
    }
    return out;
  }

  private scale(name: string, value: number): number {
    if (!Number.isFinite(value)) return 0;
    if (this.logScaled.has(name)) {
      return Math.log1p(Math.abs(value)) * (value >= 0 ? 1 : -1);
    }
    if (this.div255.has(name)) return value / 255;
    return value;
  }
}

/** Linear interpolation over the exported isotonic knots. */
export function applyCalibration(
  raw: number,
  knots: { x: number[]; y: number[] } | null | undefined,
): number {
  if (!knots || !knots.x.length) return raw;
  const { x, y } = knots;
  if (raw <= x[0]) return y[0];
  if (raw >= x[x.length - 1]) return y[y.length - 1];
  let lo = 0;
  let hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= raw) lo = mid; else hi = mid;
  }
  const span = x[hi] - x[lo];
  const t = span === 0 ? 0 : (raw - x[lo]) / span;
  return Math.min(1, Math.max(0, y[lo] + t * (y[hi] - y[lo])));
}

/**
 * Fail loudly if the spec and the ONNX graph disagree about input width.
 * A silent mismatch here produces confident nonsense.
 */
export function assertFeatureOrder(names: string[], modelInputWidth: number): void {
  if (names.length !== modelInputWidth) {
    throw new Error(
      `feature-order mismatch: the spec lists ${names.length} features but the ` +
      `ONNX model expects ${modelInputWidth}. Re-export with ` +
      `scripts/05_export_onnx.py so the spec and the model come from the same build.`,
    );
  }
}
