"""
Main entry point for LayoutLM Document Processing Service.
"""

import argparse
import sys

from config import get_config
from infrastructure.logger_utils import setup_logging, get_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LayoutLMv3 Document Processing Service"
    )

    parser.add_argument(
        "--mode",
        choices=["api", "ui", "all"],
        default="ui",
        help="Run mode: api (FastAPI), ui (Gradio), or all"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to"
    )

    args = parser.parse_args()

    # Setup logging
    config = get_config()
    setup_logging(level=config.log.level, format_type=config.log.format)
    logger = get_logger(__name__)

    logger.info(f"Starting LayoutLM service in {args.mode} mode")

    if args.mode == "api":
        from api.server import run_server
        run_server()

    elif args.mode == "ui":
        from ui.gradio_app import run_ui
        run_ui()

    elif args.mode == "all":
        import multiprocessing

        # Run API in separate process
        api_process = multiprocessing.Process(
            target=run_api_server
        )
        api_process.start()

        # Run UI in main process
        from ui.gradio_app import run_ui
        run_ui()

        # Wait for API process
        api_process.join()


def run_api_server():
    """Run API server (for multiprocessing)."""
    from api.server import run_server
    run_server()


if __name__ == "__main__":
    main()
