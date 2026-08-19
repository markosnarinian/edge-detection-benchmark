#!/usr/bin/env python3
"""Interactive two-machine setup wizard for the detector benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmark import (
    CPU_RUNTIMES,
    DEFAULT_IMG_SIZES,
    MODEL_RUNTIMES,
    RFDETR_VARIANTS,
    YOLOX_VARIANTS,
    atomic_write_text,
    parse_img_sizes,
    validate_artifact_manifest,
)

PLAN_NAME = "benchmark-plan.json"
COMMON_PACKAGES = ("numpy", "pillow", "psutil")
EXPORT_GROUPS = {
    "stable": ("onnxruntime", "openvino", "ncnn"),
    "tflite": ("tflite",),
    "executorch": ("executorch",),
}


class Wizard:
    def __init__(self, *, dry_run: bool, assume_yes: bool):
        self.dry_run = dry_run
        self.assume_yes = assume_yes

    def prompt(self, message: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        if self.assume_yes:
            print(f"{message}{suffix}: {default}")
            return default
        try:
            value = input(f"{message}{suffix}: ").strip()
        except EOFError:
            raise SystemExit("Input ended before the wizard was complete.")
        return value or default

    def confirm(self, message: str, default: bool = True) -> bool:
        if self.assume_yes:
            print(f"{message}: yes")
            return True
        marker = "Y/n" if default else "y/N"
        value = self.prompt(f"{message} ({marker})").lower()
        if not value:
            return default
        return value in {"y", "yes"}

    def run(self, command: list[str], *, cwd: Path) -> bool:
        print(f"\n$ {shlex.join(command)}")
        if self.dry_run:
            return True
        try:
            return subprocess.run(command, cwd=cwd, check=False).returncode == 0
        except OSError as e:
            print(f"Could not run command: {e}", file=sys.stderr)
            return False

    def selection(
        self, message: str, allowed: tuple[str, ...], default: tuple[str, ...]
    ) -> list[str]:
        while True:
            raw = self.prompt(message, ",".join(default))
            if raw.lower() in {"none", "off", "-"}:
                return []
            values = list(
                dict.fromkeys(
                    part.strip().lower() for part in raw.split(",") if part.strip()
                )
            )
            invalid = [value for value in values if value not in allowed]
            if not invalid:
                return values
            print(
                f"Unsupported values: {', '.join(invalid)}. Choose from: {', '.join(allowed)}"
            )

    def integer(self, message: str, default: int, *, minimum: int = 1) -> int:
        while True:
            raw = self.prompt(message, str(default))
            try:
                value = int(raw)
            except ValueError:
                print("Enter a whole number.")
                continue
            if value < minimum:
                print(f"Enter a value of at least {minimum}.")
                continue
            return value


def repository_root() -> Path:
    return Path(__file__).resolve().parent


def python_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uv_install_command(python: Path, *packages: str) -> list[str]:
    return ["uv", "pip", "install", "--python", str(python), *packages]


def prepare_environment(
    wizard: Wizard,
    *,
    label: str,
    work_dir: Path,
    base_python: str,
    packages: list[str],
) -> Path | None:
    venv = work_dir / "wizard-envs" / label
    python = python_in_venv(venv)
    marker = venv / ".wizard-setup.json"
    setup_signature = {"packages": packages}
    print(f"\nPreparing isolated {label} environment at {venv}")
    if python.exists() and marker.is_file():
        try:
            if json.loads(marker.read_text()) == setup_signature:
                print("Reusing the existing prepared environment.")
                return python
        except (OSError, json.JSONDecodeError):
            pass
    if not python.exists() and not wizard.run(
        ["uv", "venv", "--python", base_python, str(venv)], cwd=repository_root()
    ):
        return None
    if packages and not wizard.run(
        uv_install_command(python, *packages), cwd=repository_root()
    ):
        return None
    if not wizard.dry_run:
        atomic_write_text(marker, json.dumps(setup_signature, indent=2))
    return python


def benchmark_selection_args(plan: dict) -> list[str]:
    args = [
        "--yolox-variants",
        ",".join(plan["yolox_variants"]),
        "--rfdetr-variants",
        ",".join(plan["rfdetr_variants"]),
        "--img-sizes",
        ",".join(str(size) for size in plan["img_sizes"]),
        "--num-classes",
        str(plan["num_classes"]),
        "--opset",
        str(plan["opset"]),
    ]
    if not plan["yolox_variants"]:
        args.append("--skip-yolox")
    if not plan["rfdetr_variants"]:
        args.append("--skip-rfdetr")
    return args


def save_plan(plan: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(artifact_dir / PLAN_NAME, json.dumps(plan, indent=2))


def plan_error(plan: object) -> str:
    if not isinstance(plan, dict) or plan.get("schema") != 1:
        return "unsupported or missing plan schema"
    required_lists = {
        "yolox_variants": set(YOLOX_VARIANTS),
        "rfdetr_variants": set(RFDETR_VARIANTS),
        "runtimes": set(CPU_RUNTIMES),
        "img_sizes": None,
    }
    for key, allowed in required_lists.items():
        values = plan.get(key)
        if not isinstance(values, list):
            return f"{key} must be a list"
        if allowed is not None and any(
            not isinstance(value, str) or value not in allowed for value in values
        ):
            return f"{key} contains unsupported values"
    if not plan["runtimes"] or not (plan["yolox_variants"] or plan["rfdetr_variants"]):
        return "plan selects no models or runtimes"
    if not any(
        variants and runtime in MODEL_RUNTIMES[family]
        for family, variants in (
            ("yolox", plan["yolox_variants"]),
            ("rfdetr", plan["rfdetr_variants"]),
        )
        for runtime in plan["runtimes"]
    ):
        return "plan contains no supported model/runtime combination"
    if not plan["img_sizes"] or any(
        not isinstance(size, int) or size <= 0 or size % 32
        for size in plan["img_sizes"]
    ):
        return "img_sizes must contain positive multiples of 32"
    if not isinstance(plan.get("num_classes"), int) or plan["num_classes"] <= 0:
        return "num_classes must be a positive integer"
    if not isinstance(plan.get("opset"), int) or plan["opset"] <= 0:
        return "opset must be a positive integer"
    return ""


def missing_exports(plan: dict, runtimes: list[str], artifact_dir: Path) -> list[str]:
    runtime_kind = {
        "onnxruntime": "onnx",
        "openvino": "onnx",
        "ncnn": "ncnn",
        "tflite": "tflite",
        "executorch": "executorch",
    }
    extensions = {
        "onnx": ".onnx",
        "ncnn": ".param",
        "tflite": ".tflite",
        "executorch": ".pte",
    }
    requests: set[tuple[str, str, int, str]] = set()
    for variant in plan["yolox_variants"]:
        for runtime in runtimes:
            if runtime in {"onnxruntime", "openvino", "ncnn"}:
                for size in plan["img_sizes"]:
                    requests.add(
                        (f"yolox_{variant}", "yolox", size, runtime_kind[runtime])
                    )
    for variant in plan["rfdetr_variants"]:
        for runtime in runtimes:
            if runtime in {"onnxruntime", "openvino", "tflite", "executorch"}:
                for size in plan["img_sizes"]:
                    requests.add(
                        (f"rfdetr_{variant}", "rfdetr", size, runtime_kind[runtime])
                    )

    missing = []
    for name, family, size, kind in sorted(requests):
        path = artifact_dir / name / str(size) / kind / f"{name}{extensions[kind]}"
        config = {
            "model": name,
            "family": family,
            "artifact_kind": kind,
            "img_size": size,
            "num_classes": plan["num_classes"],
            "opset": plan["opset"],
        }
        valid, _, reason = validate_artifact_manifest(
            path, kind, path.with_suffix(".export.json"), config
        )
        if not valid:
            missing.append(f"{path}: {reason}")
    return missing


def print_export_problems(problems: list[str]) -> None:
    for problem in problems[:20]:
        print(f"  - {problem}")
    if len(problems) > 20:
        print(f"  - … and {len(problems) - 20} more")


def export_flow(wizard: Wizard) -> int:
    root = repository_root()
    print(
        "\nEXPORT MACHINE\n"
        "The wizard will export deployment graphs in isolated environments.\n"
        "PyTorch is not exported; its source packages/checkpoints are installed on the Pi."
    )
    yolox = wizard.selection(
        "YOLOX variants ('none' disables YOLOX)", YOLOX_VARIANTS, YOLOX_VARIANTS
    )
    rfdetr = wizard.selection(
        "RF-DETR variants ('none' disables RF-DETR)",
        tuple(RFDETR_VARIANTS),
        tuple(RFDETR_VARIANTS),
    )
    runtimes = wizard.selection("Pi runtimes", CPU_RUNTIMES, CPU_RUNTIMES)
    while True:
        raw_sizes = wizard.prompt("Square resolutions", DEFAULT_IMG_SIZES)
        sizes = parse_img_sizes(raw_sizes)
        if sizes:
            break
        print("Enter at least one positive multiple of 32.")
    num_classes = wizard.integer("Detection classes", 15)
    opset = wizard.integer("ONNX opset", 17)
    artifact_dir = (
        Path(wizard.prompt("Artifact directory", "results/artifacts"))
        .expanduser()
        .resolve()
    )
    work_dir = (
        Path(wizard.prompt("Export work directory", "work")).expanduser().resolve()
    )

    compatible = any(
        (
            family == "yolox"
            and runtime in {"pytorch", "onnxruntime", "openvino", "ncnn"}
        )
        or (
            family == "rfdetr"
            and runtime
            in {"pytorch", "onnxruntime", "openvino", "tflite", "executorch"}
        )
        for family, variants in (("yolox", yolox), ("rfdetr", rfdetr))
        if variants
        for runtime in runtimes
    )
    if not compatible:
        print(
            "The selections contain no supported model/runtime combination.",
            file=sys.stderr,
        )
        return 2

    plan = {
        "schema": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "yolox_variants": yolox,
        "rfdetr_variants": rfdetr,
        "runtimes": runtimes,
        "img_sizes": sizes,
        "num_classes": num_classes,
        "opset": opset,
        "export_groups": {},
    }
    save_plan(plan, artifact_dir)

    cells = (
        len(yolox)
        * len(
            [r for r in runtimes if r in {"pytorch", "onnxruntime", "openvino", "ncnn"}]
        )
        + len(rfdetr)
        * len(
            [
                r
                for r in runtimes
                if r in {"pytorch", "onnxruntime", "openvino", "tflite", "executorch"}
            ]
        )
    ) * len(sizes)
    print(
        f"\nPlanned benchmark: {cells} cells. Static exports can require tens of gigabytes."
    )
    if not wizard.confirm("Start the export steps?"):
        print(f"Plan saved to {artifact_dir / PLAN_NAME}; no exports were run.")
        return 0

    for group, group_members in EXPORT_GROUPS.items():
        selected = [runtime for runtime in runtimes if runtime in group_members]
        if not selected:
            continue
        if group in {"tflite", "executorch"} and not rfdetr:
            continue
        if group == "stable" and not (
            (yolox and any(r in selected for r in ("onnxruntime", "openvino", "ncnn")))
            or (rfdetr and any(r in selected for r in ("onnxruntime", "openvino")))
        ):
            continue

        default_python = (
            "python3.12"
            if group == "tflite" and shutil.which("python3.12")
            else sys.executable
        )
        if group == "tflite" and "3.12" not in Path(default_python).name:
            print(
                "TFLite export currently requires Python 3.12. Enter a Python 3.12 "
                "executable below or expect this export group to fail."
            )
        base_python = wizard.prompt(
            f"Python executable for {group} export", default_python
        )
        packages = list(COMMON_PACKAGES)
        if group == "stable":
            if any(runtime in selected for runtime in ("onnxruntime", "openvino")):
                packages.extend(("onnx", "onnxscript", "onnxsim"))
            if "ncnn" in selected:
                packages.extend(("ncnn", "pnnx"))
        python = prepare_environment(
            wizard,
            label=f"export-{group}",
            work_dir=work_dir,
            base_python=base_python,
            packages=packages,
        )
        if python is None:
            plan["export_groups"][group] = "setup_failed"
            save_plan(plan, artifact_dir)
            if not wizard.confirm(
                f"{group} setup failed. Continue with other export groups?"
            ):
                return 1
            continue

        command = [
            str(python),
            str(root / "benchmark.py"),
            "--export-only",
            "--runtimes",
            ",".join(selected),
            "--artifact-dir",
            str(artifact_dir),
            "--work-dir",
            str(work_dir),
            *benchmark_selection_args(plan),
        ]
        if group in {"tflite", "executorch"}:
            command.append("--skip-yolox")
        command_ok = wizard.run(command, cwd=root)
        missing = (
            [] if wizard.dry_run else missing_exports(plan, selected, artifact_dir)
        )
        if missing:
            print(f"{group} export is incomplete:")
            print_export_problems(missing)
        ok = command_ok and not missing
        plan["export_groups"][group] = (
            "planned" if wizard.dry_run else ("complete" if ok else "failed")
        )
        save_plan(plan, artifact_dir)
        if not ok and not wizard.confirm(
            f"{group} export failed. Continue with other groups?"
        ):
            return 1

    print("\nExport stage complete.")
    print(f"Transfer this entire directory to the Pi: {artifact_dir}")
    print(
        f"It includes {PLAN_NAME}, which the Pi stage will use to reproduce the matrix."
    )
    print(
        "Example: scp -r "
        + shlex.quote(str(artifact_dir))
        + " pi@<pi-address>:~/edge-benchmark/artifacts"
    )
    if "pytorch" in runtimes:
        print(
            "PyTorch was intentionally not bundled; the Pi stage installs it and downloads checkpoints."
        )
    return 0


def install_pi_dependencies(
    wizard: Wizard, python: Path, runtimes: list[str], root: Path
) -> bool:
    commands: list[list[str]] = [
        uv_install_command(python, *COMMON_PACKAGES)
    ]
    packages = {
        "pytorch": ["torch", "torchvision"],
        "onnxruntime": ["onnxruntime"],
        "openvino": ["openvino"],
        "ncnn": ["ncnn"],
        "tflite": ["ai-edge-litert"],
        "executorch": ["executorch"],
    }
    for runtime in runtimes:
        command = uv_install_command(python, *packages[runtime])
        if runtime == "pytorch":
            command.extend(("--default-index", "https://download.pytorch.org/whl/cpu"))
        commands.append(command)
    all_ok = True
    for command in commands:
        command_ok = wizard.run(command, cwd=root)
        if not command_ok:
            print(
                "Package installation failed. Pi OS/Python wheels vary by runtime; "
                "you may continue and the benchmark will record affected cells as failures."
            )
            if "ai-edge-litert" in command and wizard.confirm(
                "Try the tflite-runtime fallback?"
            ):
                command_ok = wizard.run(
                    uv_install_command(python, "tflite-runtime"), cwd=root
                )
            if not command_ok and not wizard.confirm("Continue setup?"):
                return False
        all_ok = command_ok and all_ok
    return all_ok


def pi_flow(wizard: Wizard, plan_argument: Path | None) -> int:
    root = repository_root()
    print(
        "\nRASPBERRY PI\n"
        "Copy the complete artifact directory from the export machine before continuing."
    )
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        print(f"Warning: this machine reports {platform.machine()}, not ARM64.")
    if plan_argument:
        plan_path = plan_argument.expanduser().resolve()
    else:
        artifact_dir = (
            Path(wizard.prompt("Transferred artifact directory", "artifacts"))
            .expanduser()
            .resolve()
        )
        plan_path = artifact_dir / PLAN_NAME
    if not plan_path.is_file():
        print(f"Plan not found: {plan_path}", file=sys.stderr)
        return 2
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read {plan_path}: {e}", file=sys.stderr)
        return 2
    invalid = plan_error(plan)
    if invalid:
        print(f"Invalid plan {plan_path}: {invalid}", file=sys.stderr)
        return 2
    artifact_dir = plan_path.parent
    print("\nImported matrix:")
    print(f"  YOLOX: {', '.join(plan['yolox_variants']) or 'disabled'}")
    print(f"  RF-DETR: {', '.join(plan['rfdetr_variants']) or 'disabled'}")
    print(f"  Runtimes: {', '.join(plan['runtimes'])}")
    print(f"  Resolutions: {', '.join(map(str, plan['img_sizes']))}")
    incomplete = [
        name
        for name, state in plan.get("export_groups", {}).items()
        if state != "complete"
    ]
    if incomplete:
        print(f"Warning: export groups not marked complete: {', '.join(incomplete)}")
    missing = (
        [] if wizard.dry_run else missing_exports(plan, plan["runtimes"], artifact_dir)
    )
    if missing:
        print("Transferred artifacts are missing, corrupt, or incompatible:")
        print_export_problems(missing)
        if not wizard.confirm(
            "Continue and record those runtime cells as failures?", default=False
        ):
            return 1

    work_dir = Path(wizard.prompt("Pi work directory", "work")).expanduser().resolve()
    output_dir = (
        Path(wizard.prompt("Results directory", "results")).expanduser().resolve()
    )
    images_raw = wizard.prompt("Local image directory (empty uses benchmark samples)")
    threads = wizard.integer("CPU threads", os.cpu_count() or 4)
    warmups = wizard.integer("Warmup runs per cell", 10, minimum=0)
    timed = wizard.integer("Timed runs per cell", 30, minimum=20)
    base_python = wizard.prompt(
        "Python executable for the Pi environment", sys.executable
    )

    python = prepare_environment(
        wizard,
        label="pi-runtime",
        work_dir=work_dir,
        base_python=base_python,
        packages=[],
    )
    if python is None:
        return 1
    runtime_marker = python.parent.parent / ".wizard-runtimes.json"
    runtime_signature = {"runtimes": plan["runtimes"]}
    dependencies_ready = False
    if runtime_marker.is_file():
        try:
            dependencies_ready = (
                json.loads(runtime_marker.read_text()) == runtime_signature
            )
        except (OSError, json.JSONDecodeError):
            pass
    if dependencies_ready:
        print("Reusing the previously installed Pi runtime packages.")
    elif wizard.confirm("Install the selected runtime packages now?"):
        dependencies_ok = install_pi_dependencies(
            wizard, python, plan["runtimes"], root
        )
        if dependencies_ok and not wizard.dry_run:
            atomic_write_text(runtime_marker, json.dumps(runtime_signature, indent=2))
        if not dependencies_ok and not wizard.confirm(
            "Some runtime packages could not be installed. Run anyway?"
        ):
            return 1

    command = [
        str(python),
        str(root / "benchmark.py"),
        "--skip-export",
        "--artifact-dir",
        str(artifact_dir),
        "--output-dir",
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--runtimes",
        ",".join(plan["runtimes"]),
        "--threads",
        str(threads),
        "--warmup-runs",
        str(warmups),
        "--timed-runs",
        str(timed),
        *benchmark_selection_args(plan),
    ]
    if images_raw:
        command.extend(("--images", str(Path(images_raw).expanduser().resolve())))
    print(
        "\nThe benchmark checkpoints after every cell. If it is interrupted, rerun "
        "this Pi wizard with the same answers to resume."
    )
    if not wizard.confirm("Run the benchmark now?"):
        print("Run this command when ready:\n" + shlex.join(command))
        return 0
    ok = wizard.run(command, cwd=root)
    if ok:
        print(f"\nBenchmark complete. Open {output_dir / 'index.html'}")
        return 0
    print(
        f"\nThe benchmark returned a failure status. Inspect {output_dir / 'index.html'} "
        "for per-cell errors; completed cells remain resumable."
    )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", nargs="?", choices=("export", "pi"), help="Stage to run"
    )
    parser.add_argument(
        "--plan", type=Path, help="Pi stage path to benchmark-plan.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Accept defaults and confirmations"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wizard = Wizard(dry_run=args.dry_run, assume_yes=args.yes)
    stage = args.stage
    if stage is None:
        choice = wizard.prompt(
            "Run which stage? (export/pi)",
            "pi" if platform.machine().lower() in {"aarch64", "arm64"} else "export",
        )
        if choice not in {"export", "pi"}:
            print("Choose 'export' or 'pi'.", file=sys.stderr)
            return 2
        stage = choice
    return export_flow(wizard) if stage == "export" else pi_flow(wizard, args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
