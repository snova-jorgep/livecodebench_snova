#!/usr/bin/env python3
"""
LiveCodeBench Report Generator

Parses LiveCodeBench evaluation results and generates unified CSV for analysis.
Uploads both summary CSV and raw output files to S3.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_s3_client():
    """Get boto3 S3 client with credentials from environment."""
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", ""),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    )


def upload_to_s3(local_filepath: str, s3_location: str) -> bool:
    """Upload a file to S3."""
    try:
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "")
        s3 = get_s3_client()
        s3.upload_file(local_filepath, bucket_name, s3_location)
        logger.info(f"Uploaded {local_filepath} to s3://{bucket_name}/{s3_location}")
        return True
    except Exception as e:
        logger.error(f"Error uploading {local_filepath} to S3: {e}")
        return False


def find_eval_all_files(run_id: str, base_dir: Path = None) -> List[Dict[str, str]]:
    """
    Find all _eval_all.json files for a given run.

    Returns:
        List of dicts with keys: provider, model, scenario, file_path
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    runs_dir = base_dir / "runs" / run_id
    if not runs_dir.exists():
        logger.error(f"Run directory not found: {runs_dir}")
        return []

    eval_files = []
    for provider_dir in runs_dir.iterdir():
        if not provider_dir.is_dir():
            continue

        provider = provider_dir.name
        output_dir = provider_dir / "output"

        if not output_dir.exists():
            continue

        # Find all _eval_all.json files
        for eval_file in output_dir.rglob("*_eval_all.json"):
            # Extract model name from directory structure
            # Path: runs/{run_id}/{provider}/output/{model}/{scenario}_{n}_{temp}_eval_all.json
            model = eval_file.parent.name

            # Extract scenario from filename
            filename = eval_file.stem  # Remove .json
            # Format: {scenario}_{n}_{temp}_eval_all or Scenario.{scenario}_{n}_{temp}_eval_all
            parts = filename.split("_")
            if len(parts) >= 4 and parts[-2] == "eval" and parts[-1] == "all":
                scenario = parts[0]  # First part is scenario
                # Remove "Scenario." prefix if present
                if scenario.startswith("Scenario."):
                    scenario = scenario.replace("Scenario.", "", 1)
            else:
                scenario = "unknown"

            eval_files.append({
                "provider": provider,
                "model": model,
                "scenario": scenario,
                "file_path": str(eval_file),
            })

    logger.info(f"Found {len(eval_files)} evaluation files for run {run_id}")
    return eval_files


def parse_eval_all_file(file_path: str) -> Optional[Dict]:
    """
    Parse a single _eval_all.json file and compute aggregate metrics.

    Returns:
        Dict with pass@1, pass@5, etc., or None if parsing fails
    """
    try:
        with open(file_path, "r") as f:
            results = json.load(f)

        if not results:
            logger.warning(f"Empty results in {file_path}")
            return None

        # Each result has pre-computed pass@1 in the file
        # Aggregate across all problems
        pass_at_1_list = [r.get("pass@1", 0.0) for r in results if "pass@1" in r]

        if not pass_at_1_list:
            logger.warning(f"No pass@1 metrics found in {file_path}")
            return None

        # Compute average pass@1
        avg_pass_1 = sum(pass_at_1_list) / len(pass_at_1_list)

        # For pass@5, we need to estimate from graded_list
        # pass@k = 1 - (1 - c/n)^k, where c = correct count, n = total samples
        # Note: Only meaningful when n >= 5
        from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k
        import numpy as np

        totals = [len(r.get("graded_list", [])) for r in results]
        corrects = [sum(r.get("graded_list", [])) for r in results]

        # Only compute pass@5 if we have enough samples (n >= 5)
        if totals and corrects and totals[0] >= 5:
            try:
                pass_5_array = estimate_pass_at_k(totals, corrects, 5)
                avg_pass_5 = float(np.mean(pass_5_array))
            except:
                avg_pass_5 = None  # Set to None if estimation fails
        else:
            avg_pass_5 = None  # Not applicable when n < 5

        # Get metadata from first result
        first_result = results[0]

        return {
            "pass_at_1": avg_pass_1,
            "pass_at_5": avg_pass_5,
            "num_problems": len(results),
            "total_samples": sum(totals) if totals else 0,
        }

    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None


def generate_summary_csv(eval_files: List[Dict], run_id: str, output_path: Path):
    """
    Generate unified summary CSV from evaluation results.

    CSV columns: date, provider, model, scenario, pass_at_1, pass_at_5, num_problems
    """
    rows = []

    for eval_info in eval_files:
        metrics = parse_eval_all_file(eval_info["file_path"])

        if metrics is None:
            continue

        row = {
            "date": run_id,
            "provider": eval_info["provider"],
            "model": eval_info["model"],
            "scenario": eval_info["scenario"],
            "pass_at_1": f"{metrics['pass_at_1']:.4f}",
            "pass_at_5": f"{metrics['pass_at_5']:.4f}" if metrics['pass_at_5'] is not None else "",
            "num_problems": metrics["num_problems"],
        }
        rows.append(row)

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        fieldnames = ["date", "provider", "model", "scenario", "pass_at_1", "pass_at_5", "num_problems"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Generated summary CSV with {len(rows)} entries: {output_path}")
    return output_path


def upload_raw_outputs(run_id: str, base_dir: Path = None):
    """
    Upload all raw output files to S3.

    Uploads entire runs/{run_id}/ directory structure to S3.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    runs_dir = base_dir / "runs" / run_id
    if not runs_dir.exists():
        logger.error(f"Run directory not found: {runs_dir}")
        return False

    success_count = 0
    fail_count = 0

    # Upload all files in the run directory (except summary CSV which is uploaded separately)
    for file_path in runs_dir.rglob("*"):
        if file_path.is_file():
            # Skip summary_results.csv - it's uploaded separately to the root location
            if file_path.name == "summary_results.csv":
                continue

            # Calculate relative path from runs_dir
            relative_path = file_path.relative_to(runs_dir)
            s3_key = f"fc-so-testing-suite/livecodebench/{run_id}/raw_outputs/{relative_path.as_posix()}"

            if upload_to_s3(str(file_path), s3_key):
                success_count += 1
            else:
                fail_count += 1

    logger.info(f"Uploaded {success_count} raw output files to S3 ({fail_count} failed)")
    return fail_count == 0


def generate_and_upload_report(run_id: str, skip_upload: bool = False, base_dir: Path = None) -> bool:
    """
    Generate report and upload to S3.

    Args:
        run_id: Run ID to process
        skip_upload: If True, skip S3 upload
        base_dir: Base directory (defaults to script directory)

    Returns:
        True if successful, False otherwise
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    # Find all evaluation files
    eval_files = find_eval_all_files(run_id, base_dir)
    if not eval_files:
        logger.error("No evaluation files found")
        return False

    # Generate summary CSV
    summary_path = base_dir / "runs" / run_id / "summary_results.csv"
    generate_summary_csv(eval_files, run_id, summary_path)

    if skip_upload:
        logger.info("Skipping S3 upload")
        return True

    # Upload summary CSV
    s3_summary_key = f"fc-so-testing-suite/livecodebench/{run_id}/summary_results.csv"
    upload_to_s3(str(summary_path), s3_summary_key)

    # Upload raw outputs
    upload_raw_outputs(run_id, base_dir)

    logger.info(f"Report generation and upload complete for run {run_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate report and upload LiveCodeBench results to S3",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID (timestamp) to process. If not provided, will use the latest run.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Generate CSV only, skip S3 upload",
    )

    args = parser.parse_args()

    # Find run_id if not provided
    if args.run_id:
        run_id = args.run_id
    else:
        # Find latest run
        runs_dir = Path(__file__).parent / "runs"
        if not runs_dir.exists():
            logger.error("No runs directory found")
            sys.exit(1)

        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            logger.error("No runs found")
            sys.exit(1)

        run_id = run_dirs[-1].name
        logger.info(f"Using latest run: {run_id}")

    # Generate and upload report
    success = generate_and_upload_report(run_id, args.skip_upload)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
