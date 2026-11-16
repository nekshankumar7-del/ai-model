# Using the models in the Angular app

Everything here runs in the browser. There is no server call and no external
AI API: the ONNX models execute under `onnxruntime-web`, OCR runs on WASM, and
the uploaded document never leaves the page.

## What the Python export gives you

Run `python scripts/05_export_onnx.py` and copy `angular/assets/factfind/`
into your app's assets folder:

| file | what it is |
| --- | --- |
| `factfind-spec.json` | every rule the browser needs: field definitions read from the knowledge file, feature names and their order, scaling, calibration knots, fill thresholds, charsets, regexes |
| `linker.onnx` | re-ranks field candidates for a line |
| `confidence.onnx` | P(this extracted value is correct) |
| `det.onnx`, `rec.onnx`, `cls.onnx` | PP-OCRv4 text detection and recognition, if you use the ONNX OCR path |

`factfind-spec.json` is the contract. The TypeScript never hard-codes a field,
a label or a threshold - change `knowledge/fact-find-knowledge.json`, re-export,
and the browser picks the change up with no code change.

## Install

```bash
npm install onnxruntime-web
npm install tesseract.js          # the simplest OCR path
npm install pdfjs-dist            # only if you accept PDFs
```

In `angular.json`, make sure the WASM binaries are served:

```json
{
  "glob": "**/*.wasm",
  "input": "node_modules/onnxruntime-web/dist/",
  "output": "assets/ort/"
}
```

## Wire it up

```ts
import { Component, inject } from '@angular/core';
import { FactFindExtractionService } from './factfind/fact-find-extraction.service';
import { TesseractOcrReader } from './factfind/ocr/tesseract-ocr-reader';

@Component({
  selector: 'app-fill-from-document',
  standalone: true,
  template: `
    <input type="file" multiple
           accept=".jpg,.jpeg,.png,.pdf"
           (change)="onFiles($event)" />
    <p *ngIf="busy">{{ progress }}</p>
  `,
})
export class FillFromDocumentComponent {
  private readonly extraction = inject(FactFindExtractionService);
  busy = false;
  progress = '';

  async onFiles(event: Event) {
    const files = (event.target as HTMLInputElement).files;
    if (!files?.length) return;

    this.busy = true;
    const reader = new TesseractOcrReader();
    await reader.init();
    this.extraction.ocr = reader;

    const { perFile, merged, conflicts } = await this.extraction.extractFiles(files, {
      caseType: 'mortgage',
      onProgress: (done, total, label) => {
        this.progress = `Reading ${label} (${done + 1} of ${total})`;
      },
    });

    // Only values the pipeline is confident about.
    this.factFindForm.patchValue(merged as any);

    // Everything it refused to guess, for the adviser to confirm.
    const needsReview = perFile
      .flatMap(r => r.fields)
      .filter(f => f.status === 'unsure');

    this.busy = false;
  }
}
```

## The three statuses

| status | meaning | what to do with it |
| --- | --- | --- |
| `filled` | the calibrated probability cleared the threshold for that field's kind | patch it into the form |
| `unsure` | something was read, but not confidently enough | show `candidates` (top 3) and let the adviser pick |
| `missing` | nothing on the page looked like this field | leave it alone |

This split is the whole point. The system is tuned so that a wrong fill costs
far more than an unsure one, and it will refuse rather than guess. It will
never fill an NI number, email address or phone number that does not satisfy
that field's pattern, whatever the model thinks.

`thresholdShift` on `ExtractionOptions` moves every threshold at once: positive
fills less, negative fills more. Use it for a "be cautious" toggle, not as a
substitute for re-fitting the thresholds on real data.

## Choosing an OCR reader

The service takes any `OcrReader`. Two paths:

**tesseract.js** (`ocr/tesseract-ocr-reader.ts`) - one npm package, works
immediately, strong on printed forms, weaker on cursive handwriting.

**PP-OCRv4 under onnxruntime-web** - uses the `det.onnx` / `rec.onnx` models
exported alongside, and is what the shipped thresholds were fitted against.
It needs the DB detection post-process and CTC decoding implemented against
those two graphs; that work is not included here.

This matters for correctness, not just accuracy: **the thresholds in
`factfind-spec.json` were calibrated against PP-OCRv4 confidences.** If you
ship tesseract.js, re-run the Python training with `FACTFIND_OCR=tesseract` so
the calibration matches the engine actually producing the confidences.
Otherwise the probabilities will be systematically optimistic and the
wrong-fill rate will be higher than the reports say.

## Performance

The budget is 4 s per page on a mid-range laptop. The model inference is
negligible (two small tree ensembles over a few hundred rows). Essentially all
the time is OCR, so:

- run extraction in a web worker so the UI stays responsive;
- keep the prepared page at or below 2200 px on the long side (the
  preprocessor already caps this);
- for multi-page PDFs, process pages in sequence and stream results.

## Files

```
angular/src/factfind/
  spec.types.ts                     types for factfind-spec.json
  text.ts                           normalisation + fuzzy ratios (mirrors rapidfuzz)
  parsers.ts                        per-kind value parsers
  matcher.ts                        segmentation + field matching
  features.ts                       model feature vectors + calibration
  preprocess.ts                     canvas preprocessing + quality metrics
  fact-find-extraction.service.ts   the orchestrator
  ocr/tesseract-ocr-reader.ts       OCR adapter
```

Each TypeScript file mirrors the Python module of the same name. If you change
one, change both, and re-run `tests/test_onnx_parity.py`.
