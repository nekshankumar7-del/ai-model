"""End-to-end test of the public-dataset HTR pipeline on generated crops.

No network, no downloaded corpus: the point is to prove the machinery works -
adapter to manifest, validation, writer-independent split, Unicode charset,
augmentation, training step, CTC decode, ONNX export and parity - so that a
failure on real data is a data problem rather than a plumbing problem.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from htr import splits as split_module                       # noqa: E402
from htr.augment import AugmentConfig, augment               # noqa: E402
from htr.charset import UnicodeCharset, build_charset, script_of  # noqa: E402
from htr.manifest import Sample, normalise_text, validate    # noqa: E402
from htr.models import TORCH_AVAILABLE, build_model, minimum_width  # noqa: E402
from htr.preprocess import PreprocessConfig, normalise       # noqa: E402

WORDS = ["Salary", "Income", "Pension", "Mortgage", "Yes", "No", "45000",
         "क्ष", "नमस्ते", "१२३"]


def _render(text: str, rng: random.Random) -> np.ndarray:
    """A crude but deterministic 'handwriting' crop: text drawn with PIL.

    The font is deliberately large. At PIL's default 11px the glyphs survive
    the resize to 32px tall as a few grey smears, and the overfit test below
    then measures whether CTC can read a smear rather than whether the
    pipeline is wired up.
    """
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=26)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    width = int(probe.textlength(text, font=font)) + rng.randint(8, 20)
    image = Image.new("L", (max(28, width), 48), rng.randint(235, 255))
    draw = ImageDraw.Draw(image)
    draw.text((5, 8), text, font=font, fill=rng.randint(10, 70))
    return np.array(image)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("htr_corpus")
    import cv2
    rng = random.Random(7)
    samples = []
    for writer in range(6):
        for index, word in enumerate(WORDS):
            image = _render(word, rng)
            path = directory / f"w{writer}_{index}.png"
            cv2.imwrite(str(path), image)
            samples.append(Sample(
                id=f"fake/w{writer}/{index}", text=word, dataset="fake",
                level="word", script=script_of(word[0]), language="en",
                writer=f"fake-{writer}", image=str(path),
            ))
    return samples


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def test_text_normalisation_is_nfc_and_whitespace_safe():
    assert normalise_text("  à   b  ") == "à b"
    assert normalise_text("﻿Yes") == "Yes"


def test_charset_covers_every_training_label(corpus):
    charset, report = build_charset([s.text for s in corpus], min_count=1)
    assert all(charset.supports(s.text) for s in corpus)
    assert report["samples_made_unencodable"] == 0
    assert "Devanagari" in charset.scripts()
    assert charset.decode_indices(charset.encode("नमस्ते")) == "नमस्ते"


def test_charset_drops_rare_characters_and_says_so(corpus):
    charset, report = build_charset([s.text for s in corpus], min_count=1000)
    assert report["dropped_characters"] > 0
    assert len(charset) == 0


def test_preprocess_gives_fixed_height_and_dark_ink(corpus):
    config = PreprocessConfig(height=32)
    import cv2
    for sample in corpus[:5]:
        image = cv2.imread(sample.image, cv2.IMREAD_GRAYSCALE)
        out = normalise(image, config)
        assert out.shape[0] == 32
        assert out.ndim == 2
        border = np.concatenate([out[0], out[-1]])
        assert float(np.median(border)) > 100, "ink should be dark on light"


def test_preprocess_inverts_a_white_on_black_crop():
    image = np.zeros((40, 80), np.uint8)
    image[12:28, 10:70] = 255
    out = normalise(image, PreprocessConfig(height=32))
    assert float(np.median(np.concatenate([out[0], out[-1]]))) > 128


def test_augmentation_preserves_shape_and_dtype(corpus):
    import cv2
    rng = random.Random(1)
    image = cv2.imread(corpus[0].image, cv2.IMREAD_GRAYSCALE)
    for strength in (0.5, 1.0, 1.5):
        out = augment(image, rng, AugmentConfig(strength=strength,
                                                probability=1.0))
        assert out.dtype == np.uint8
        assert out.ndim == 2
        assert out.shape[0] == image.shape[0]


def test_validation_drops_empty_and_duplicate_rows(corpus):
    rows = list(corpus) + [
        Sample(id="fake/empty", text="   ", dataset="fake", image=corpus[0].image),
        Sample(id="fake/dupe", text=corpus[0].text, dataset="fake",
               image=corpus[0].image),
    ]
    kept, report = validate(rows, dedupe=True)
    assert report.empty_label == 1
    assert report.duplicates == 1
    assert len(kept) == len(corpus)


def test_split_is_writer_independent(corpus):
    parts, report = split_module.assign(corpus, validation=0.2, test=0.2,
                                        min_writers_for_grouping=4)
    assert report.per_dataset["fake"]["writer_independent"] is True
    assert split_module.check_leakage(parts)["clean"]
    writers = {side: {s.writer for s in rows} for side, rows in parts.items()}
    assert not (writers["train"] & writers["test"])


def test_split_falls_back_when_there_are_no_writer_ids():
    rows = [Sample(id=f"x/{i}", text="a", dataset="x", writer="unknown")
            for i in range(50)]
    _parts, report = split_module.assign(rows)
    assert report.per_dataset["x"]["writer_independent"] is False
    assert "within-writer" in report.per_dataset["x"]["note"]


def test_minimum_width_leaves_room_for_repeated_characters():
    # "aa" needs a blank between the two a's, so more frames than "ab".
    assert minimum_width(2) >= 3 * 4


# --------------------------------------------------------------------------
# Training, decoding and export
# --------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_training_step_runs_and_loss_is_finite(corpus):
    import torch
    from htr.data import ManifestDataset, make_loader

    charset, _ = build_charset([s.text for s in corpus], min_count=1)
    dataset = ManifestDataset(corpus, charset, augment=None)
    loader = make_loader(dataset, batch_size=8, shuffle=False, workers=0)
    model = build_model("crnn", charset.n_classes, hidden=48)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    images, targets, lengths = next(iter(loader))
    log_probs = model(images)
    input_lengths = torch.full((images.size(0),), log_probs.size(0),
                               dtype=torch.long)
    loss = ctc(log_probs, targets, input_lengths, lengths)
    assert torch.isfinite(loss), "a finite CTC loss means the crops fit"
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_model_overfits_a_tiny_set(corpus):
    """The strongest cheap signal that the whole path is wired correctly."""
    import torch
    from htr.data import ManifestDataset, make_loader
    from htr.evaluate import evaluate

    subset = [s for s in corpus if s.writer == "fake-0"]
    charset, _ = build_charset([s.text for s in subset], min_count=1)
    dataset = ManifestDataset(subset, charset, augment=None)
    loader = make_loader(dataset, batch_size=len(subset), shuffle=False,
                         workers=0, bucket=False)
    model = build_model("crnn", charset.n_classes, hidden=64)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-3)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    # CTC sits in an all-blank minimum for the first couple of hundred steps
    # before the alignment breaks open; stopping earlier tests nothing.
    for _ in range(400):
        for images, targets, lengths in loader:
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, lengths)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    metrics = evaluate(model, subset, charset, batch_size=16)
    assert metrics["cer"] < 0.35, f"CER {metrics['cer']:.3f} on data it memorised"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_ctc_decode_collapses_repeats_and_blanks():
    import torch
    from htr.evaluate import greedy_decode

    charset = UnicodeCharset(("a", "b"))
    # frames: a a blank a b b  ->  "aab"
    path = [1, 1, 0, 1, 2, 2]
    log_probs = torch.full((len(path), 1, charset.n_classes), -10.0)
    for t, index in enumerate(path):
        log_probs[t, 0, index] = 0.0
    assert greedy_decode(log_probs.log_softmax(2), charset)[0].text == "aab"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_onnx_export_is_width_dynamic_and_matches_torch(tmp_path, corpus):
    from htr.export import export_bundle

    charset, _ = build_charset([s.text for s in corpus], min_count=1)
    model = build_model("crnn", charset.n_classes, hidden=48)
    bundle = export_bundle(model, charset, tmp_path, name="test-htr")

    assert bundle["parity"]["passed"]
    assert bundle["parity"]["max_abs_difference"] < 2e-4
    assert {r["width"] for r in bundle["parity"]["per_width"]} == {128, 256, 384}
    loaded = UnicodeCharset.load(tmp_path / "test-htr-charset.json")
    assert loaded.characters == charset.characters


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_checkpoint_round_trips(tmp_path, corpus):
    import torch
    from htr.train import TrainConfig, load_checkpoint, save_checkpoint

    charset, _ = build_charset([s.text for s in corpus], min_count=1)
    config = TrainConfig(architecture="crnn", hidden=48, level="word")
    model = build_model("crnn", charset.n_classes, hidden=48)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, charset, config, {})

    restored, restored_charset, restored_config, _ = load_checkpoint(path)
    assert restored_charset.characters == charset.characters
    assert restored_config.level == "word"
    sample = torch.randn(1, 1, 32, 128)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(sample), restored(sample), atol=1e-6)


# --------------------------------------------------------------------------
# Registry hygiene
# --------------------------------------------------------------------------

def test_every_registered_dataset_declares_a_licence_and_an_adapter():
    from htr.adapters import ADAPTERS
    from htr import registry

    for spec in registry.ALL:
        assert spec.licence and spec.licence_url, spec.id
        assert spec.adapter in ADAPTERS, spec.id
        if spec.access == "auto":
            assert spec.resources, f"{spec.id} is auto but has no URL"
        else:
            assert spec.manual_instructions, f"{spec.id} needs instructions"


def test_registry_ids_are_unique():
    from htr import registry
    ids = [s.id for s in registry.ALL]
    assert len(ids) == len(set(ids))


def test_archive_extraction_rejects_a_traversing_member(tmp_path):
    import zipfile
    from htr.extraction import UnsafeArchive, extract_one

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(UnsafeArchive):
        extract_one(archive, tmp_path / "out")
