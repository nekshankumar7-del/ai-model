/**
 * Gradient-boosted tree evaluator.
 *
 * The linker and the confidence model are small tree ensembles. They ship as
 * JSON rather than ONNX, which means the browser needs no inference runtime
 * for them at all - evaluating one is a walk down a binary tree.
 *
 * This reproduces sklearn's HistGradientBoostingClassifier arithmetic exactly:
 * a raw score of `baseline + sum of the leaf reached in every tree`, then a
 * logistic link. Verified against sklearn to within 1e-17 by the Python
 * parity test, so the browser and the training code agree to the last bit.
 */

export interface TreeJson {
  isLeaf: boolean[];
  value: number[];
  featureIdx: number[];
  threshold: number[];
  left: number[];
  right: number[];
  missingGoToLeft: boolean[];
}

export interface TreeModelJson {
  type: 'histGradientBoostingBinary';
  baseline: number;
  nFeatures: number;
  nTrees: number;
  link: 'logistic';
  trees: TreeJson[];
}

export class TreeModel {
  constructor(private readonly model: TreeModelJson) {
    if (model.type !== 'histGradientBoostingBinary') {
      throw new Error(`unsupported tree model type: ${model.type}`);
    }
  }

  static async load(url: string): Promise<TreeModel> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`could not load ${url} (${response.status})`);
    }
    return new TreeModel((await response.json()) as TreeModelJson);
  }

  get nFeatures(): number {
    return this.model.nFeatures;
  }

  /** P(positive class) for one feature vector. */
  predict(features: Float32Array | number[]): number {
    if (features.length !== this.model.nFeatures) {
      throw new Error(
        `feature-count mismatch: the model expects ${this.model.nFeatures} ` +
        `features but got ${features.length}. Re-export the spec and the ` +
        `model together.`,
      );
    }
    let raw = this.model.baseline;
    for (const tree of this.model.trees) {
      raw += this.leafValue(tree, features);
    }
    return 1 / (1 + Math.exp(-raw));
  }

  /** P(positive class) for a batch, as a Float32Array. */
  predictBatch(rows: Array<Float32Array | number[]>): Float32Array {
    const out = new Float32Array(rows.length);
    for (let i = 0; i < rows.length; i++) out[i] = this.predict(rows[i]);
    return out;
  }

  private leafValue(tree: TreeJson, features: Float32Array | number[]): number {
    let node = 0;
    // Guard against a malformed file sending us round forever.
    for (let steps = 0; steps < 1024; steps++) {
      if (tree.isLeaf[node]) return tree.value[node];
      const x = features[tree.featureIdx[node]];
      if (Number.isNaN(x)) {
        node = tree.missingGoToLeft[node] ? tree.left[node] : tree.right[node];
      } else {
        node = x <= tree.threshold[node] ? tree.left[node] : tree.right[node];
      }
    }
    throw new Error('tree walk did not terminate; the model file is malformed');
  }
}
