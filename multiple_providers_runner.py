#!/usr/bin/env python3
"""
Multi-provider runner for LiveCodeBench.

Runs LiveCodeBench across multiple providers in parallel using subprocess execution.
Each provider runs with its own API keys and base URL via environment variables.
"""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_api_key_env_var(provider: str) -> str:
    """Get the environment variable name for a provider's API key."""
    provider_key_map = {
        "sambanova": "SAMBANOVA_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "together": "TOGETHER_API_KEY",
        "novita": "NOVITA_API_KEY",
    }
    return provider_key_map.get(provider.lower(), f"{provider.upper()}_API_KEY")


def build_command(
    provider: str,
    model_name: str,
    scenario: str,
    config: dict,
    debug: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Build the LiveCodeBench command and environment variables.

    Returns:
        Tuple of (command_list, env_dict)
    """
    benchmark_config = config["benchmark"]
    gen_config = config["generation"]

    # Base command
    cmd = [
        sys.executable,
        "-m",
        "lcb_runner.runner.main",
        "--model", model_name,
        "--scenario", scenario,
        "--release_version", benchmark_config["release_version"],
        "--evaluate",
        "--n", str(gen_config["n"]),
        "--temperature", str(gen_config["temperature"]),
        "--max_tokens", str(gen_config["max_tokens"]),
        "--top_p", str(gen_config["top_p"]),
        "--openai_timeout", str(gen_config["timeout"]),
    ]

    # Add debug mode (runs only first 15 test cases)
    if debug:
        cmd.append("--debug")

    # Environment variables
    api_key_var = get_api_key_env_var(provider)
    base_url = config["provider_urls"].get(provider.lower())

    if not base_url:
        raise ValueError(f"Base URL not found for provider: {provider}")

    # Validate API key is set (for both dry-run and actual execution)
    api_key = os.getenv(api_key_var)
    if not api_key:
        raise ValueError(f"API key not found: {api_key_var} environment variable not set")

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"] = api_key
    env["_API_KEY_VAR"] = api_key_var  # Store for logging

    return cmd, env


def run_model(
    provider: str,
    canonical_name: str,
    model_name: str,
    scenario: str,
    config: dict,
    run_id: str,
    dry_run: bool,
    debug: bool = False,
) -> Dict:
    """
    Run LiveCodeBench for a single model.

    Returns:
        Dict with execution results
    """
    try:
        cmd, env = build_command(provider, model_name, scenario, config, debug)

        # Create working directory for this provider run
        work_dir = Path(f"runs/{run_id}/{provider}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # API key validation happens in build_command() for both dry-run and actual execution
        if dry_run:
            cmd_str = " ".join(cmd)
            api_key_var = env.get("_API_KEY_VAR", "API_KEY")
            logger.info(f"[DRY-RUN] {provider} / {canonical_name}:")
            logger.info(f"  WORK_DIR: {work_dir}")
            logger.info(f"  ENV: OPENAI_API_KEY=${api_key_var} OPENAI_BASE_URL={env['OPENAI_BASE_URL']}")
            logger.info(f"  CMD: {cmd_str}\n")
            return {
                "provider": provider,
                "canonical_name": canonical_name,
                "model_name": model_name,
                "scenario": scenario,
                "status": "dry_run",
                "command": cmd_str,
                "work_dir": str(work_dir),
            }

        # Execute command
        logger.info(f"Starting: {provider} / {canonical_name} ({scenario})")
        result = subprocess.run(
            cmd,
            env=env,
            cwd=work_dir,  # Run in provider-specific directory
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info(f"Completed: {provider} / {canonical_name}")
            status = "success"
        else:
            logger.error(f"Failed: {provider} / {canonical_name}")
            logger.error(f"Error output: {result.stderr[:500]}")
            status = "failed"

        return {
            "provider": provider,
            "canonical_name": canonical_name,
            "model_name": model_name,
            "scenario": scenario,
            "status": status,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ Timeout: {provider} / {canonical_name}")
        return {
            "provider": provider,
            "canonical_name": canonical_name,
            "model_name": model_name,
            "scenario": scenario,
            "status": "timeout",
        }
    except Exception as e:
        logger.error(f"Exception: {provider} / {canonical_name} - {e}")
        return {
            "provider": provider,
            "canonical_name": canonical_name,
            "model_name": model_name,
            "scenario": scenario,
            "status": "error",
            "error": str(e),
        }


def run_provider(
    provider: str,
    models: Dict[str, str],
    scenarios: List[str],
    config: dict,
    run_id: str,
    dry_run: bool,
    debug: bool = False,
) -> List[Dict]:
    """
    Run all models for a single provider in parallel.

    Uses ThreadPoolExecutor to run multiple models concurrently per provider.
    """
    results = []
    models_per_provider = config["parallelism"]["models_per_provider"]

    # Filter out "not_available" models
    available_models = {
        canonical: provider_name
        for canonical, provider_name in models.items()
        if provider_name != "not_available"
    }

    if not available_models:
        logger.warning(f"No available models for provider: {provider}")
        return results

    logger.info(f"Provider {provider}: Running {len(available_models)} models with {models_per_provider} parallel workers")

    # Run models in parallel within this provider
    with ThreadPoolExecutor(max_workers=models_per_provider) as executor:
        futures = []

        for canonical_name, model_name in available_models.items():
            for scenario in scenarios:
                future = executor.submit(
                    run_model,
                    provider,
                    canonical_name,
                    model_name,
                    scenario,
                    config,
                    run_id,
                    dry_run,
                    debug,
                )
                futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


def run_all_providers(
    config: dict,
    run_id: str,
    dry_run: bool = False,
    debug: bool = False,
) -> List[Dict]:
    """
    Run LiveCodeBench across all providers in parallel.

    Uses ThreadPoolExecutor to run multiple providers concurrently.
    """
    model_mappings = config["model_mappings"]
    scenarios = config["benchmark"]["scenarios"]
    num_providers = config["parallelism"]["providers"]

    logger.info(f"Running {len(model_mappings)} providers with {num_providers} parallel workers")
    logger.info(f"Scenarios: {', '.join(scenarios)}")

    all_results = []

    # Run providers in parallel using ThreadPoolExecutor (simpler than multiprocessing for our use case)
    with ThreadPoolExecutor(max_workers=num_providers) as executor:
        futures = {}

        for provider, models in model_mappings.items():
            future = executor.submit(
                run_provider,
                provider,
                models,
                scenarios,
                config,
                run_id,
                dry_run,
                debug,
            )
            futures[future] = provider

        for future in as_completed(futures):
            provider = futures[future]
            try:
                results = future.result()
                all_results.extend(results)

                # Log provider summary
                successes = sum(1 for r in results if r["status"] == "success")
                logger.info(f"Provider {provider} completed: {successes}/{len(results)} successful")
            except Exception as e:
                logger.error(f"Provider {provider} failed with exception: {e}")

    return all_results


def print_summary(results: List[Dict]):
    """Print execution summary."""
    if not results:
        logger.info("No results to summarize")
        return

    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    error = sum(1 for r in results if r["status"] == "error")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    dry_run = sum(1 for r in results if r["status"] == "dry_run")

    logger.info("\n" + "="*60)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total runs:    {total}")
    if dry_run > 0:
        logger.info(f"Dry runs:      {dry_run}")
    else:
        logger.info(f"Success:    {success}")
        logger.info(f"Failed:     {failed}")
        logger.info(f"Error:      {error}")
        logger.info(f"Timeout:    {timeout}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run LiveCodeBench across multiple providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only first 15 test cases per model)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    if args.dry_run:
        logger.info("DRY-RUN MODE: Commands will be printed but not executed")

    # Debug mode: CLI flag overrides config setting
    debug_mode = args.debug or config.get("benchmark", {}).get("debug", False)
    if debug_mode:
        logger.info("DEBUG MODE: Running only first 15 test cases per model")

    # Generate run ID (timestamp)
    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"Run ID: {run_id}")

    # Run all providers
    start_time = datetime.now()
    results = run_all_providers(
        config=config,
        run_id=run_id,
        dry_run=args.dry_run,
        debug=debug_mode,
    )
    end_time = datetime.now()

    # Print summary
    print_summary(results)
    logger.info(f"Total execution time: {end_time - start_time}")

    # Generate report and upload to S3 (skip upload in debug or dry-run mode)
    if not args.dry_run:
        logger.info("\n" + "="*60)
        logger.info("GENERATING REPORT AND UPLOADING TO S3")
        logger.info("="*60)

        from livecodebench_report import generate_and_upload_report

        skip_upload = debug_mode  # Skip S3 upload in debug mode
        if skip_upload:
            logger.info("Debug mode enabled - skipping S3 upload")

        success = generate_and_upload_report(run_id, skip_upload=skip_upload)
        if not success:
            logger.error("Report generation failed")
            sys.exit(1)

    # Exit with error if any runs failed
    if not args.dry_run:
        failed_count = sum(1 for r in results if r["status"] in ["failed", "error", "timeout"])
        if failed_count > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
