"""Download a trained model from the project's GitHub release, check it against the
release's SHA256SUMS, and unpack it into play_tests/models/<name>/ - where every script
here looks for models. Standard library only: works before uv sync.

    python tools/fetch_model.py                  # b20c256, the deployed network
    python tools/fetch_model.py b15c192latest --tag models-2026-10

The release files are <name>.zip and SHA256SUMS (made by tools/package_models.py).
--base-url fetches from somewhere else (any URL urllib can open, file:// included).
"""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "robdawg-dev/NeuralZ"
DEFAULT_TAG = "models-2026-10"  # update when a new model release is published


def download(url, path):
    with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, base_url, root=ROOT):
    """Download, verify and unpack one model; returns its folder."""
    with tempfile.TemporaryDirectory() as tmp:
        sums_path = os.path.join(tmp, "SHA256SUMS")
        download(base_url + "/SHA256SUMS", sums_path)
        with open(sums_path) as f:
            sums = dict(reversed(line.split()) for line in f if line.strip())
        zname = name + ".zip"
        if zname not in sums:
            raise SystemExit("fetch_model: {} is not in this release ({})".format(
                name, ", ".join(sorted(z[:-4] for z in sums))))
        zpath = os.path.join(tmp, zname)
        print("downloading {} ...".format(base_url + "/" + zname), flush=True)
        download(base_url + "/" + zname, zpath)
        if sha256(zpath) != sums[zname]:
            raise SystemExit("fetch_model: checksum mismatch for {} - not unpacked".format(zname))
        prefix = "play_tests/models/{}/".format(name)
        with zipfile.ZipFile(zpath) as z:
            members = z.namelist()
            bad = [m for m in members if not m.startswith(prefix) or ".." in m.split("/")]
            if bad:
                raise SystemExit("fetch_model: unexpected paths in {}: {}".format(zname, bad))
            z.extractall(root)
    folder = os.path.join(root, "play_tests", "models", name)
    print("{} ready in {}".format(name, folder))
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", default="b20c256")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--base-url", help="default: the GitHub release of --tag")
    args = parser.parse_args(argv)
    base = (args.base_url or "https://github.com/{}/releases/download/{}".format(
        REPO, args.tag)).rstrip("/")
    try:
        fetch(args.name, base)
    except OSError as e:
        sys.exit("fetch_model: {} ({})".format(e, base))


if __name__ == "__main__":
    main()
