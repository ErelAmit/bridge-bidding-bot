from pathlib import Path

from scipy.io import loadmat


def inspect_single_mat(mat_path: Path) -> None:
    print(f"\n=== Inspecting {mat_path.name} ===")
    if not mat_path.exists():
        print("File does NOT exist at that path")
        return

    mat = loadmat(mat_path)
    print("Keys inside the .mat file:")
    for key in mat.keys():
        if key.startswith("__"):
            continue
        value = mat[key]
        print(f"  {key}: type={type(value)}, shape={getattr(value, 'shape', None)}")


def main() -> None:
    print("Starting inspect_mat.py")

    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root is: {project_root}")

    data_dir = project_root / "data"

    inspect_single_mat(data_dir / "data_train.mat")
    inspect_single_mat(data_dir / "cost_train.mat")


if __name__ == "__main__":
    main()
