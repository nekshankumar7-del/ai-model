/**
 * Page preprocessing in the browser - the canvas twin of preprocess.py.
 *
 * Gets from a phone photo or PDF page to a deskewed, illumination-flattened
 * greyscale canvas the OCR reader can work on, and measures the image-quality
 * features the confidence model consumes.
 *
 * Every step is defensive: if it cannot find what it is looking for it leaves
 * the image alone rather than mangling it. That matters more than squeezing
 * out the last bit of quality, because a mangled page produces confident
 * nonsense rather than an honest refusal.
 *
 * PDFs need pdfjs-dist; it is imported lazily so apps that only accept images
 * never pay for it.
 */
import { PageQuality } from './spec.types';

export interface PreparedPage {
  /** Greyscale, flattened, deskewed - what the OCR reader is given. */
  canvas: HTMLCanvasElement;
  /** The colour page at the same geometry, for showing the user. */
  display: HTMLCanvasElement;
  quality: PageQuality;
  pageIndex: number;
}

const MAX_SIDE = 2200;
const MIN_SIDE = 600;

export class PagePreprocessor {
  /** Turn one uploaded file into prepared pages. */
  async loadFile(file: File): Promise<PreparedPage[]> {
    const images = file.type === 'application/pdf' || /\.pdf$/i.test(file.name)
      ? await this.renderPdf(file)
      : [await this.loadImage(file)];
    return images.map((canvas, i) => this.preparePage(canvas, i));
  }

  private async loadImage(file: File): Promise<HTMLCanvasElement> {
    const bitmap = await createImageBitmap(await file.arrayBuffer()
      .then((buffer) => new Blob([buffer], { type: file.type || 'image/jpeg' })));
    const { width, height } = fitTo(bitmap.width, bitmap.height);
    const canvas = makeCanvas(width, height);
    canvas.getContext('2d')!.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    return canvas;
  }

  private async renderPdf(file: File): Promise<HTMLCanvasElement[]> {
    const pdfjs = await import('pdfjs-dist');
    const data = new Uint8Array(await file.arrayBuffer());
    const doc = await pdfjs.getDocument({ data }).promise;
    const out: HTMLCanvasElement[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const base = page.getViewport({ scale: 1 });
      const scale = Math.min(MAX_SIDE / Math.max(base.width, base.height), 3);
      const viewport = page.getViewport({ scale });
      const canvas = makeCanvas(Math.round(viewport.width), Math.round(viewport.height));
      await page.render({
        canvas,
        canvasContext: canvas.getContext('2d')!,
        viewport,
      } as Parameters<typeof page.render>[0]).promise;
      out.push(canvas);
    }
    return out;
  }

  // ------------------------------------------------------------ pipeline
  preparePage(source: HTMLCanvasElement, pageIndex: number): PreparedPage {
    const sized = resizeToWork(source);
    const angle = estimateSkew(sized);
    const display = Math.abs(angle) >= 0.25 && Math.abs(angle) <= 15
      ? rotate(sized, angle)
      : sized;

    const grey = toGreyscale(display);
    const flattened = flattenIllumination(grey);
    const quality = describeQuality(flattened, angle, pageIndex);

    return { canvas: flattened, display, quality, pageIndex };
  }
}

// -------------------------------------------------------------- helpers

function makeCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, width);
  canvas.height = Math.max(1, height);
  return canvas;
}

function fitTo(width: number, height: number): { width: number; height: number } {
  const longest = Math.max(width, height);
  let scale = 1;
  if (longest > MAX_SIDE) scale = MAX_SIDE / longest;
  else if (longest < MIN_SIDE) scale = MIN_SIDE / longest;
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

function resizeToWork(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const { width, height } = fitTo(canvas.width, canvas.height);
  if (width === canvas.width && height === canvas.height) return canvas;
  const out = makeCanvas(width, height);
  const ctx = out.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, width, height);
  return out;
}

function rotate(canvas: HTMLCanvasElement, degrees: number): HTMLCanvasElement {
  const out = makeCanvas(canvas.width, canvas.height);
  const ctx = out.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate((-degrees * Math.PI) / 180);
  ctx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
  return out;
}

function toGreyscale(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const out = makeCanvas(canvas.width, canvas.height);
  const src = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
  const dst = out.getContext('2d')!.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < src.data.length; i += 4) {
    const g = (src.data[i] * 299 + src.data[i + 1] * 587 + src.data[i + 2] * 114) / 1000;
    dst.data[i] = dst.data[i + 1] = dst.data[i + 2] = g;
    dst.data[i + 3] = 255;
  }
  out.getContext('2d')!.putImageData(dst, 0, 0);
  return out;
}

/**
 * Remove the soft shadow gradient a phone photo leaves behind, by dividing
 * the image by a heavily blurred copy of itself.
 */
function flattenIllumination(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext('2d')!;
  const image = ctx.getImageData(0, 0, width, height);

  const grey = new Float32Array(width * height);
  for (let i = 0, p = 0; i < image.data.length; i += 4, p++) grey[p] = image.data[i];

  const radius = Math.max(8, Math.round(Math.min(width, height) / 40));
  const background = boxBlur(grey, width, height, radius);

  const out = makeCanvas(width, height);
  const dst = out.getContext('2d')!.createImageData(width, height);
  for (let p = 0, i = 0; p < grey.length; p++, i += 4) {
    const bg = background[p] || 1;
    const value = Math.max(0, Math.min(255, (grey[p] / bg) * 255));
    dst.data[i] = dst.data[i + 1] = dst.data[i + 2] = value;
    dst.data[i + 3] = 255;
  }
  out.getContext('2d')!.putImageData(dst, 0, 0);
  return out;
}

/** Separable box blur, run twice to approximate a Gaussian. */
function boxBlur(src: Float32Array, width: number, height: number,
                 radius: number): Float32Array {
  let current = src;
  for (let pass = 0; pass < 2; pass++) {
    current = blurAxis(current, width, height, radius, true);
    current = blurAxis(current, width, height, radius, false);
  }
  return current;
}

function blurAxis(src: Float32Array, width: number, height: number,
                  radius: number, horizontal: boolean): Float32Array {
  const out = new Float32Array(src.length);
  const outer = horizontal ? height : width;
  const inner = horizontal ? width : height;
  const window = radius * 2 + 1;

  for (let o = 0; o < outer; o++) {
    const index = (i: number) => (horizontal ? o * width + i : i * width + o);
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      sum += src[index(Math.min(inner - 1, Math.max(0, i)))];
    }
    for (let i = 0; i < inner; i++) {
      out[index(i)] = sum / window;
      const leaving = src[index(Math.min(inner - 1, Math.max(0, i - radius)))];
      const entering = src[index(Math.min(inner - 1, Math.max(0, i + radius + 1)))];
      sum += entering - leaving;
    }
  }
  return out;
}

/**
 * Skew angle from the dominant near-horizontal edges, by projection profile:
 * the rotation whose row-sum profile has the sharpest peaks is the one that
 * lines the text up.
 */
function estimateSkew(canvas: HTMLCanvasElement): number {
  const scale = Math.min(1, 700 / Math.max(canvas.width, canvas.height));
  const w = Math.max(1, Math.round(canvas.width * scale));
  const h = Math.max(1, Math.round(canvas.height * scale));
  const small = makeCanvas(w, h);
  small.getContext('2d')!.drawImage(canvas, 0, 0, w, h);
  const data = small.getContext('2d')!.getImageData(0, 0, w, h).data;

  const ink = new Uint8Array(w * h);
  let sum = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const g = (data[i] * 299 + data[i + 1] * 587 + data[i + 2] * 114) / 1000;
    sum += g;
    ink[p] = g < 160 ? 1 : 0;
  }
  const mean = sum / (w * h);
  if (mean > 250 || mean < 5) return 0;          // blank or black page

  let bestAngle = 0;
  let bestScore = -1;
  for (let angle = -8; angle <= 8; angle += 0.5) {
    const score = projectionScore(ink, w, h, angle);
    if (score > bestScore) {
      bestScore = score;
      bestAngle = angle;
    }
  }
  return bestAngle;
}

function projectionScore(ink: Uint8Array, w: number, h: number, angle: number): number {
  const slope = Math.tan((angle * Math.PI) / 180);
  const rows = new Float64Array(h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!ink[y * w + x]) continue;
      const shifted = Math.round(y - (x - w / 2) * slope);
      if (shifted >= 0 && shifted < h) rows[shifted]++;
    }
  }
  // Sharper peaks mean better-aligned text; variance of the profile captures it.
  let total = 0;
  for (const v of rows) total += v;
  const mean = total / h;
  let variance = 0;
  for (const v of rows) variance += (v - mean) ** 2;
  return variance;
}

function describeQuality(canvas: HTMLCanvasElement, skewAngle: number,
                         _pageIndex: number): PageQuality {
  const width = canvas.width;
  const height = canvas.height;
  const data = canvas.getContext('2d')!.getImageData(0, 0, width, height).data;

  let sum = 0;
  let sumSquares = 0;
  let ink = 0;
  const n = width * height;
  const grey = new Float32Array(n);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const g = data[i];
    grey[p] = g;
    sum += g;
    sumSquares += g * g;
    if (g < 128) ink++;
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSquares / n - mean * mean);

  // Variance of the Laplacian: the standard sharpness proxy.
  let lapSum = 0;
  let lapSumSq = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = y * width + x;
      const lap =
        4 * grey[p] - grey[p - 1] - grey[p + 1] - grey[p - width] - grey[p + width];
      lapSum += lap;
      lapSumSq += lap * lap;
      count++;
    }
  }
  const lapMean = count ? lapSum / count : 0;
  const blur = count ? Math.max(0, lapSumSq / count - lapMean * lapMean) : 0;

  const background = boxBlur(grey, width, height,
    Math.max(8, Math.round(Math.min(width, height) / 40)));
  let bgSum = 0;
  let bgSumSq = 0;
  for (const v of background) { bgSum += v; bgSumSq += v * v; }
  const bgMean = bgSum / n;
  const shadowSpread = Math.sqrt(Math.max(0, bgSumSq / n - bgMean * bgMean));

  return {
    width, height,
    blur_score: blur,
    contrast: Math.sqrt(variance),
    brightness: mean,
    ink_ratio: ink / n,
    skew_angle: skewAngle,
    // The browser pipeline does not attempt the paper-quad crop, so this is
    // reported as 0 rather than guessed at.
    page_found: 0,
    shadow_spread: shadowSpread,
  };
}
