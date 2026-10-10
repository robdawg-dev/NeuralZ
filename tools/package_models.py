"""Package trained models for a GitHub release (MODEL_RELEASE_PLAN.md): one zip per model
holding exactly the files play_tests/policy_loading.py loads (model JSON, weights, and any
metadata*.json), under play_tests/models/<name>/ - so unzipping at the repo root puts the
model in place - plus SHA256SUMS and a NOTES.md draft for the release text.

    uv run python tools/package_models.py [b20c256 b15c192latest ...] [--out dist/release]

Then create the release by hand or with gh, e.g.
    gh release create models-2026-10 dist/release/*.zip dist/release/SHA256SUMS \\
        --title "Trained models (October 2026)" --notes-file dist/release/NOTES.md
and set fetch_model.py's DEFAULT_TAG to the new tag. Standard library only.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_MODELS = ["b20c256", "b15c192latest", "b10c128mb1024", "2016net"]


def model_specs():
    """MODEL_SPECS from play_tests/policy_loading.py, read without importing TensorFlow."""
    path = os.path.join(ROOT, "play_tests", "policy_loading.py")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    start = source.index("MODEL_SPECS = {")
    end = source.index("\n}\n", start) + 2
    namespace = {}
    exec(source[start:end], namespace)  # a plain dict literal
    return namespace["MODEL_SPECS"]


def files_for(name, specs, models_dir):
    spec = specs[name]
    folder = os.path.join(models_dir, name)
    files = [os.path.join(folder, spec["json"]), os.path.join(folder, spec["weights"])]
    files += sorted(glob.glob(os.path.join(folder, "metadata*.json")))
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("{}: missing {}".format(name, ", ".join(missing)))
    return files


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def notes_for(name, files):
    """A short description from the model's metadata, if it has any."""
    meta = [f for f in files if os.path.basename(f).startswith("metadata")]
    text = "- **{}**: {} MB".format(name, round(sum(os.path.getsize(f) for f in files) / 1e6))
    if meta:
        with open(meta[0]) as f:
            epochs = json.load(f).get("epochs", [])
        if epochs:
            best = max(epochs, key=lambda e: e.get("val_accuracy", 0))
            text += "; {} epochs; held-out top-1 {:.1%}".format(
                len(epochs), best.get("val_accuracy", 0))
            if "val_top5_accuracy" in best:
                text += " / top-5 {:.1%}".format(best["val_top5_accuracy"])
    return text


def package(names, out, models_dir=None):
    models_dir = models_dir or os.path.join(ROOT, "play_tests", "models")
    specs = model_specs()
    os.makedirs(out, exist_ok=True)
    sums, notes = [], ["# Trained models", "",
                       "Unzip at the repository root (or run `python tools/fetch_model.py "
                       "<name>`); each lands in `play_tests/models/<name>/`.", ""]
    for name in names:
        files = files_for(name, specs, models_dir)
        zpath = os.path.join(out, name + ".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, "play_tests/models/{}/{}".format(name, os.path.basename(f)))
        sums.append("{}  {}".format(sha256(zpath), os.path.basename(zpath)))
        notes.append(notes_for(name, files))
        print("{}: {} files, {:.0f} MB".format(zpath, len(files), os.path.getsize(zpath) / 1e6))
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    with open(os.path.join(out, "NOTES.md"), "w") as f:
        f.write("\n".join(notes) + "\n")
    return sums


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--out", default=os.path.join(ROOT, "dist", "release"))
    args = parser.parse_args(argv)
    package(args.models, args.out)
    print("SHA256SUMS and NOTES.md in {}".format(args.out))


if __name__ == "__main__":
    main()
