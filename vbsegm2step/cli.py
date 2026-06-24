"""Command-line interface for VBSegm2Step."""

import sys
import argparse
import glob
import os
from pathlib import Path
from typing import List

from .config import Config
from .pipeline import VBSegm2StepPipeline
from .utils import same_path


def build_config(model601: Path = None, model602: Path = None,
                 model601_variant: str = None) -> Config:
    """Build config using defaults, env vars, then explicit CLI overrides."""
    return Config.from_overrides(
        model601=model601,
        model602=model602,
        model601_variant=model601_variant,
    )


def _matched_input_files(input_dir: Path, pattern: str) -> List[Path]:
    """Return sorted batch inputs without printing or loading models."""
    pattern_path = Path(input_dir) / pattern
    return sorted(Path(path) for path in glob.glob(str(pattern_path), recursive=True))


def batch(input_dir: Path, output_dir: Path, pattern: str = '*.nii.gz',
          model601: Path = None, model602: Path = None,
          model601_variant: str = None):
    """Process multiple CT scans in batch mode.
    
    Args:
        input_dir: Directory containing input CT scans
        output_dir: Directory for output segmentations
        pattern: File pattern to match
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    input_files = _matched_input_files(input_dir, pattern)
    if not input_files:
        print(f"❌ No files found for pattern {pattern!r} in {input_dir}")
        sys.exit(1)

    # Create config
    print("Initializing Pipeline for batch processing...")
    config = build_config(
        model601=model601,
        model602=model602,
        model601_variant=model601_variant,
    )
    print(str(config))
    
    try:
        pipeline = VBSegm2StepPipeline(config)
        
        # Process directory
        success_count = pipeline.process_directory(input_dir, output_dir, pattern)
        
        if success_count == 0:
            print(f"❌ Batch processing failed: 0/{len(input_files)} files successful")
            sys.exit(1)
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


def predict(input_file: Path, output_file: Path, model601: Path = None,
            model602: Path = None, model601_variant: str = None):
    """Process a single CT scan file.

    Args:
        input_file: Path to input CT scan (e.g., .nii.gz)
        output_file: Path to output segmentation file (.nii.gz)
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    if same_path(input_file, output_file):
        print("❌ Input and output paths cannot be the same")
        sys.exit(1)

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    # Create output parent directory if needed
    if output_file.parent:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create config
    print("Initializing Pipeline for single-file prediction...")
    config = build_config(
        model601=model601,
        model602=model602,
        model601_variant=model601_variant,
    )
    print(str(config))

    try:
        pipeline = VBSegm2StepPipeline(config)

        result = pipeline.process_file(input_file, output_file)
        if result is False:
            print("❌ Prediction failed")
            sys.exit(1)
        print(f"✅ Prediction completed: {output_file}")

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


def validate(model601: Path = None, model602: Path = None,
             model601_variant: str = None):
    """Validate model paths and configuration."""
    # Create config
    config = build_config(
        model601=model601,
        model602=model602,
        model601_variant=model601_variant,
    )
    print("Validating configuration...")
    print(str(config))
    
    # Check model paths
    if config.validate_model_paths():
        print("✅ All model paths are valid")
    else:
        print("❌ Some model paths are invalid. Use `downloadmodels` command to download models.")
        sys.exit(1)
    
    # Test model initialization
    try:
        print("Testing model initialization...")
        pipeline = VBSegm2StepPipeline(config)
        print("✅ Models initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        sys.exit(1)

def _download_model_specs(config: Config, all_model601_variants: bool = False):
    if not all_model601_variants:
        return [
            ("nnU-Net 601", config.HF_NNUNET601, config.PATH_NNUNET601),
            ("nnU-Net 602", config.HF_NNUNET602, config.PATH_NNUNET602),
        ]

    specs = []
    for variant in sorted(Config.MODEL601_VARIANTS):
        variant_config = Config.from_overrides(model601_variant=variant)
        specs.append((
            f"nnU-Net 601 ({variant})",
            variant_config.HF_NNUNET601,
            variant_config.PATH_NNUNET601,
        ))
    specs.append(("nnU-Net 602", config.HF_NNUNET602, config.PATH_NNUNET602))
    return specs


def _validate_downloaded_models(config: Config, all_model601_variants: bool = False) -> bool:
    if not all_model601_variants:
        return config.validate_model_paths()

    ok = True
    for variant in sorted(Config.MODEL601_VARIANTS):
        variant_config = Config.from_overrides(
            model602=config.PATH_NNUNET602,
            model601_variant=variant,
        )
        ok = variant_config.validate_model_paths() and ok
    return ok


def download_models(model601: Path = None, model602: Path = None,
                    model601_variant: str = None,
                    all_model601_variants: bool = False):
    """Download model weights defined in the config via Hugging Face.

    Uses the Hugging Face repo IDs and local paths from `Config`.
    Only runs when invoked via the CLI command `downloadmodels`.
    """
    if all_model601_variants and (model601 is not None or os.environ.get(Config.MODEL601_ENV)):
        print(
            f"❌ --all-model601-variants cannot be combined with --model601 "
            f"or ${Config.MODEL601_ENV}; each released variant has its own configured path."
        )
        sys.exit(1)

    config = build_config(
        model601=model601,
        model602=model602,
        model601_variant=model601_variant,
    )
    print("Downloading models using Hugging Face Hub...")
    print(str(config))

    from huggingface_hub import snapshot_download

    models = _download_model_specs(config, all_model601_variants=all_model601_variants)

    any_failed = False
    for title, repo_id, local_dir in models:
        local_dir = Path(local_dir)
        print(f"→ {title}: {repo_id} → {local_dir}")
        try:
            # Ensure destination directory exists
            local_dir.mkdir(parents=True, exist_ok=True)

            # Download full repository snapshot into the specified directory
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                ignore_patterns=[
                    "**/model_best*",
                    "**/logs*",
                    "**/*.zip",
                    "*.zip",
                    "**/*.md",
                    "*.md",
                    "**/*.txt",
                    "*.txt",
                    "**/*.png",
                    "*.png",
                    "suppl/**",
                ]
            )
            print(f"✅ Downloaded to {local_dir} \n")
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user during download")
            sys.exit(1)
        except Exception as e:
            any_failed = True
            print(f"❌ Failed to download {title} from {repo_id}: {e}")

    # Final validation of paths
    if not any_failed and _validate_downloaded_models(
        config,
        all_model601_variants=all_model601_variants,
    ):
        print("✅ All model paths present after download")
    elif any_failed:
        print("⚠️ Some downloads failed. Please check errors above.")
        sys.exit(1)
    else:
        print("⚠️ Download finished, but paths still invalid. Check configuration.")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="VBSegm2Step: Vertebral Body Segmentation Pipeline"
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    def add_model_args(subparser):
        subparser.add_argument(
            '--model601-variant',
            choices=sorted(Config.MODEL601_VARIANTS),
            default=None,
            help=(
                f'Task 601 released model variant. Overrides ${Config.MODEL601_VARIANT_ENV}; '
                f'default: {Config.DEFAULT_MODEL601_VARIANT}.'
            ),
        )
        subparser.add_argument(
            '--model601',
            type=Path,
            default=None,
            help=f'Model 601 root directory. Overrides ${Config.MODEL601_ENV} and config defaults.',
        )
        subparser.add_argument(
            '--model602',
            type=Path,
            default=None,
            help=f'Model 602 root directory. Overrides ${Config.MODEL602_ENV} and config defaults.',
        )
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple CT scans in batch mode')
    batch_parser.add_argument('-i', '--input_dir', type=Path, required=True, help='Directory containing input CT scans')
    batch_parser.add_argument('-o', '--output_dir', type=Path, required=True, help='Directory for output segmentations')
    batch_parser.add_argument('-p', '--pattern', default='*.nii.gz', help='File pattern to match')
    add_model_args(batch_parser)

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model paths and configuration')
    add_model_args(validate_parser)

    # Download command
    download_parser = subparsers.add_parser('downloadmodels', help='Download model weights')
    add_model_args(download_parser)
    download_parser.add_argument(
        '--all-model601-variants',
        action='store_true',
        help='Download both released Task 601 variants, ResEncL and ResEncM, plus Task 602.',
    )
    
    # Predict single file command
    predict_parser = subparsers.add_parser('predict', help='Predict a single CT scan file')
    predict_parser.add_argument('-i', '--input', dest='input_file', type=Path, required=True, help='Input CT scan file path')
    predict_parser.add_argument('-o', '--output', dest='output_file', type=Path, required=True, help='Output segmentation file path')
    add_model_args(predict_parser)

    try:
        args = parser.parse_args()

        if args.command == 'batch':
            batch(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                model601=args.model601,
                model602=args.model602,
                model601_variant=args.model601_variant,
            )
        elif args.command == 'predict':
            predict(
                input_file=args.input_file,
                output_file=args.output_file,
                model601=args.model601,
                model602=args.model602,
                model601_variant=args.model601_variant,
            )
        elif args.command == 'validate':
            validate(
                model601=args.model601,
                model602=args.model602,
                model601_variant=args.model601_variant,
            )
        elif args.command == 'downloadmodels':
            download_models(
                model601=args.model601,
                model602=args.model602,
                model601_variant=args.model601_variant,
                all_model601_variants=args.all_model601_variants,
            )
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
