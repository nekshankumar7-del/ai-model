/**
 * OCR reader backed by tesseract.js.
 *
 * This is the recommended starting point: one npm package, runs entirely in
 * the browser on WASM, no server and no external API, and it gives per-line
 * boxes and confidences, which is exactly what the pipeline needs.
 *
 *   npm install tesseract.js
 *
 *   const reader = new TesseractOcrReader();
 *   await reader.init();              // downloads the language data once
 *   service.ocr = reader;
 *
 * Accuracy note: tesseract is strong on printed text and weaker on cursive
 * handwriting than the PP-OCRv4 models exported alongside this code. The
 * thresholds shipped in factfind-spec.json were fitted against PP-OCRv4
 * output, so if you run tesseract in production, re-fit them by pointing the
 * Python training run at the same engine (FACTFIND_OCR=tesseract) - otherwise
 * the calibrated probabilities will be optimistic.
 */
import type { OcrLine, OcrReader } from '../spec.types';

type TesseractWorker = {
  recognize: (image: HTMLCanvasElement) => Promise<{ data: TesseractData }>;
  terminate: () => Promise<unknown>;
};

interface TesseractData {
  lines?: Array<{
    text: string;
    confidence: number;
    bbox: { x0: number; y0: number; x1: number; y1: number };
  }>;
}

export class TesseractOcrReader implements OcrReader {
  private worker?: TesseractWorker;

  constructor(
    private readonly language = 'eng',
    /** Where tesseract.js should fetch its core and language data from. */
    private readonly options: Record<string, unknown> = {},
  ) {}

  async init(): Promise<void> {
    if (this.worker) return;
    const { createWorker } = await import('tesseract.js');
    this.worker = (await createWorker(this.language, 1, this.options)) as unknown as TesseractWorker;
  }

  async read(image: HTMLCanvasElement, pageIndex: number): Promise<OcrLine[]> {
    await this.init();
    const { data } = await this.worker!.recognize(image);
    const lines: OcrLine[] = [];

    (data.lines ?? []).forEach((line, index) => {
      const text = (line.text ?? '').trim();
      if (!text) return;
      const { x0, y0, x1, y1 } = line.bbox;
      lines.push({
        text,
        box: [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)],
        // tesseract reports 0..100.
        confidence: Math.max(0, Math.min(1, (line.confidence ?? 0) / 100)),
        pageWidth: image.width,
        pageHeight: image.height,
        lineIndex: index,
        pageIndex,
      });
    });

    return mergeSplitLines(lines);
  }

  async dispose(): Promise<void> {
    await this.worker?.terminate();
    this.worker = undefined;
  }
}

/**
 * Join fragments the detector split mid-line. A label and its value are often
 * detected as separate boxes, and the matcher works on whole lines.
 */
export function mergeSplitLines(
  lines: OcrLine[], yTolerance = 0.6, xGapTextHeights = 2.5,
): OcrLine[] {
  if (!lines.length) return [];
  const heights = lines.map((l) => l.box[3]).sort((a, b) => a - b);
  const medianHeight = heights[Math.floor(heights.length / 2)] || 1;

  const centre = (l: OcrLine) => l.box[1] + l.box[3] / 2;
  const right = (l: OcrLine) => l.box[0] + l.box[2];
  const bottom = (l: OcrLine) => l.box[1] + l.box[3];

  const sorted = [...lines].sort((a, b) => centre(a) - centre(b) || a.box[0] - b.box[0]);
  const merged: OcrLine[] = [];

  for (const line of sorted) {
    let placed = false;
    for (const target of merged) {
      if (Math.abs(centre(line) - centre(target)) > yTolerance * medianHeight) continue;
      const gap = line.box[0] >= right(target)
        ? line.box[0] - right(target)
        : target.box[0] - right(line);
      if (gap > xGapTextHeights * medianHeight) continue;

      const [left, rightMost] = target.box[0] <= line.box[0] ? [target, line] : [line, target];
      const x0 = Math.min(target.box[0], line.box[0]);
      const y0 = Math.min(target.box[1], line.box[1]);
      const x1 = Math.max(right(target), right(line));
      const y1 = Math.max(bottom(target), bottom(line));
      target.text = `${left.text} ${rightMost.text}`.trim();
      target.box = [x0, y0, x1 - x0, y1 - y0];
      target.confidence = Math.min(target.confidence, line.confidence);
      placed = true;
      break;
    }
    if (!placed) merged.push({ ...line });
  }

  merged.sort((a, b) =>
    Math.round(centre(a) / medianHeight) - Math.round(centre(b) / medianHeight)
    || a.box[0] - b.box[0]);
  merged.forEach((line, i) => { line.lineIndex = i; });
  return merged;
}
