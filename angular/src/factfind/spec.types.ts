/**
 * Types for factfind-spec.json, the file the Python export writes.
 *
 * The spec is the contract between the trained models and this code: feature
 * names and their order, the scaling applied to them, the calibration knots,
 * the fill thresholds, and every field rule read from the knowledge file.
 * Nothing here is hard-coded per field - edit the knowledge file, re-export,
 * and this code picks the change up.
 */

export type FieldKind =
  | 'text' | 'name' | 'fullName' | 'date' | 'choice' | 'yesNo'
  | 'number' | 'money' | 'email' | 'phone' | 'niNumber';

export type FieldStatus = 'filled' | 'unsure' | 'missing';

export interface ChoiceSpec {
  value: string;
  label: string;
  /** Every normalised string a writer might put down for this choice. */
  surfaceForms: string[];
}

export interface OnlyIfSpec {
  key: string;
  equals?: unknown;
  filled?: boolean;
}

export interface FieldSpec {
  /** 'personal-details.fullName' */
  key: string;
  section: string;
  field: string;
  label: string;
  kind: FieldKind;
  aliases: string[];
  /** Normalised label + aliases, longest first. */
  labelForms: string[];
  required: boolean;
  when?: string;
  onlyIf?: OnlyIfSpec;
  min?: number;
  max?: number;
  minAge?: number;
  vocabulary?: string[];
  allowOther?: boolean;
  choices?: ChoiceSpec[];
}

export interface SectionSpec {
  id: string;
  title: string;
  headings: string[];
  default: boolean;
  when: string | null;
  save: Record<string, unknown>;
  repeating: boolean;
}

export interface ModelSpec {
  file: string;
  kept: boolean;
  family?: string | null;
  blend?: number | null;
  featureNames: string[];
  inputName: string;
  outputName: string;
  positiveClassIndex: number;
  calibration?: { x: number[]; y: number[]; method: string } | null;
}

export interface FactFindSpec {
  specVersion: number;
  pipelineVersion: string;
  knowledgeVersion: number;
  generated: string;
  preprocessing: {
    maxSide: number;
    minSide: number;
    steps: string[];
    adaptiveThreshold: { blockSize: number; C: number; method: string };
    deskew: { maxAngleDegrees: number; minAngleDegrees: number };
    ruledLineRemoval: { horizontalDivisor: number; verticalDivisor: number };
  };
  ocr: {
    engine: string;
    models: { detection: string; recognition: string; classification: string };
    textScoreFloor: number;
    lineMerge: { yTolerance: number; xGapTextHeights: number };
  };
  segmentation: { separators: string[]; maxLabelWords: number };
  matching: {
    labelScoreFloor: number;
    assignMinScore: number;
    weights: { label: number; parse: number; split: number };
    sectionBonus: {
      sameSection: number; defaultSection: number;
      otherSection: number; suppressed: number;
    };
    invalidParsePenalty: number;
  };
  parsing: {
    blankTokens: string[];
    yesForms: string[];
    noForms: string[];
    charsets: Record<string, string>;
    niNumberPattern: string;
    emailPattern: string;
    ukPhone: { digits: number; leadingZero: boolean; mobilePrefix: string };
    choiceSnapThreshold: number;
    choiceMarginThreshold: number;
    vocabularySnapThreshold: number;
  };
  features: {
    linker: { featureNames: string[]; kindOrder: string[]; separatorOrder: string[] };
    confidence: { featureNames: string[] };
    scaling: { log1pSigned: string[]; divide255: string[]; note: string };
  };
  models: { linker: ModelSpec; confidence: ModelSpec };
  decision: {
    thresholds: Record<string, number>;
    hardFloors: Record<string, number>;
    statuses: FieldStatus[];
    unsureTopK: number;
    note: string;
  };
  targets: { wrongFillByKind: Record<string, number> };
  choicesFrom: Record<string, string[]>;
  fields: FieldSpec[];
  sections: SectionSpec[];
  otherHeadings: string[];
  artefacts: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

/** One line of text read off a page by whichever OCR engine is wired in. */
export interface OcrLine {
  text: string;
  /** x, y, width, height in pixels of the prepared page. */
  box: [number, number, number, number];
  /** 0..1 */
  confidence: number;
  pageWidth: number;
  pageHeight: number;
  lineIndex: number;
  pageIndex: number;
}

/** Image-quality features the confidence model consumes. */
export interface PageQuality {
  width: number;
  height: number;
  blur_score: number;
  contrast: number;
  brightness: number;
  ink_ratio: number;
  skew_angle: number;
  page_found: number;
  shadow_spread: number;
}

/** Anything that can turn a page image into lines of text. */
export interface OcrReader {
  read(image: ImageData | HTMLCanvasElement, pageIndex: number): Promise<OcrLine[]>;
}

export interface ParseResult {
  value: unknown;
  parseScore: number;
  valid: boolean;
  candidates: Array<[unknown, number]>;
  reason: string;
  raw: string;
}

export interface FieldPrediction {
  /** 'dependants[0].name' */
  key: string;
  /** 'dependants.name' */
  field: string;
  kind: FieldKind;
  label: string;
  value: unknown;
  status: FieldStatus;
  confidence: number;
  candidates: unknown[];
  raw: string;
  page: number;
  box: [number, number, number, number] | null;
  reason: string;
}

export interface ExtractionResult {
  source: string;
  /** Grouped the way the CRM saves it: section id -> values (or rows). */
  sections: Record<string, unknown>;
  fields: FieldPrediction[];
  summary: { filled: number; unsure: number; pages: number };
  warnings: string[];
}
