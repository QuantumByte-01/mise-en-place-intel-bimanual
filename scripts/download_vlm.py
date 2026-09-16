"""Download only the files needed for the compact SmolVLM planner."""

from huggingface_hub import snapshot_download

from mise_en_place import config as C


def main() -> None:
    path = snapshot_download(
        repo_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        revision="067788b187b95ebe7b2e040b3e4299e342e5b8fd",
        local_dir=C.REPO_ROOT / "checkpoints" / "smolvlm2-256m",
        allow_patterns=[
            "*.json",
            "*.txt",
            "model.safetensors",
        ],
    )
    print(f"downloaded SmolVLM planner to {path}")


if __name__ == "__main__":
    main()
