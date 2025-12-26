"""
Training process management with subprocess handling.
Provides async training execution with progress callbacks.
"""

import subprocess
import threading
import time
import os
from typing import Optional, Callable
from queue import Queue, Empty

from .base import TrainingBackend, TrainingConfig, TrainingProgress, TrainingStatus


class TrainingProcess:
    """Manages a training subprocess with progress tracking."""

    _active_process: Optional['TrainingProcess'] = None

    def __init__(
        self,
        backend: TrainingBackend,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ):
        """Initialize training process.

        Args:
            backend: Training backend to use
            config: Training configuration
            progress_callback: Function called with progress updates
        """
        self.backend = backend
        self.config = config
        self.progress_callback = progress_callback

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._progress = TrainingProgress()
        self._output_queue = Queue()
        self._stop_requested = False
        self._start_time = 0

    @classmethod
    def get_running_process(cls) -> Optional['TrainingProcess']:
        """Get currently running training process."""
        return cls._active_process

    @property
    def is_running(self) -> bool:
        """Check if training is currently running."""
        return (self._process is not None and
                self._process.poll() is None)

    @property
    def progress(self) -> TrainingProgress:
        """Get current progress."""
        return self._progress

    def start(self) -> bool:
        """Start training process.

        Returns:
            True if started successfully
        """
        if self.is_running:
            return False

        # Validate data first
        is_valid, error = self.backend.validate_data(self.config.data_path)
        if not is_valid:
            self._progress.status = TrainingStatus.FAILED
            self._progress.error = error
            return False

        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)

        # Build command
        try:
            cmd = self.backend.get_command(self.config)
        except Exception as e:
            self._progress.status = TrainingStatus.FAILED
            self._progress.error = str(e)
            return False

        # Start process
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.backend.get_install_path()
            )
        except Exception as e:
            self._progress.status = TrainingStatus.FAILED
            self._progress.error = f"Failed to start process: {e}"
            return False

        self._start_time = time.time()
        self._stop_requested = False
        self._progress.status = TrainingStatus.RUNNING
        self._progress.total_iterations = self.config.iterations

        # Start output reading thread
        self._thread = threading.Thread(target=self._read_output, daemon=True)
        self._thread.start()

        # Register as active process
        TrainingProcess._active_process = self

        return True

    def stop(self) -> None:
        """Stop training process."""
        self._stop_requested = True

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._progress.status = TrainingStatus.CANCELLED

        if TrainingProcess._active_process == self:
            TrainingProcess._active_process = None

    def _read_output(self) -> None:
        """Read process output in background thread."""
        try:
            for line in self._process.stdout:
                if self._stop_requested:
                    break

                # Parse progress
                parsed = self.backend.parse_output(line)
                if parsed:
                    self._progress.iteration = parsed.iteration or self._progress.iteration
                    self._progress.loss = parsed.loss or self._progress.loss
                    self._progress.psnr = parsed.psnr or self._progress.psnr
                    self._progress.message = parsed.message
                    self._progress.elapsed_seconds = time.time() - self._start_time

                    # Calculate ETA
                    if self._progress.iteration > 0:
                        rate = self._progress.iteration / self._progress.elapsed_seconds
                        remaining = self._progress.total_iterations - self._progress.iteration
                        self._progress.eta_seconds = remaining / rate if rate > 0 else 0

                    if parsed.error:
                        self._progress.error = parsed.error

                    if parsed.status == TrainingStatus.COMPLETED:
                        self._progress.status = TrainingStatus.COMPLETED

                    # Notify callback
                    if self.progress_callback:
                        self.progress_callback(self._progress)

                # Store in queue for retrieval
                self._output_queue.put(line)

        except Exception as e:
            self._progress.error = str(e)
            self._progress.status = TrainingStatus.FAILED

        finally:
            # Wait for process to finish
            if self._process:
                return_code = self._process.wait()

                if return_code != 0 and self._progress.status == TrainingStatus.RUNNING:
                    self._progress.status = TrainingStatus.FAILED
                    self._progress.error = f"Process exited with code {return_code}"
                elif self._progress.status == TrainingStatus.RUNNING:
                    self._progress.status = TrainingStatus.COMPLETED

            if TrainingProcess._active_process == self:
                TrainingProcess._active_process = None

            # Final callback
            if self.progress_callback:
                self.progress_callback(self._progress)

    def get_output_lines(self, max_lines: int = 100) -> list:
        """Get recent output lines.

        Args:
            max_lines: Maximum lines to return

        Returns:
            List of output lines
        """
        lines = []
        try:
            while len(lines) < max_lines:
                lines.append(self._output_queue.get_nowait())
        except Empty:
            pass
        return lines

    def get_result_path(self) -> Optional[str]:
        """Get path to training result.

        Returns:
            Path to trained model or None
        """
        if self._progress.status != TrainingStatus.COMPLETED:
            return None
        return self.backend.get_final_model_path(self.config.output_path)


def get_running_process() -> Optional[TrainingProcess]:
    """Get currently running training process."""
    return TrainingProcess.get_running_process()


def start_training(
    backend: TrainingBackend,
    config: TrainingConfig,
    progress_callback: Optional[Callable[[TrainingProgress], None]] = None
) -> TrainingProcess:
    """Start a new training process.

    Args:
        backend: Training backend to use
        config: Training configuration
        progress_callback: Progress update callback

    Returns:
        TrainingProcess instance
    """
    process = TrainingProcess(backend, config, progress_callback)
    process.start()
    return process


def stop_training() -> None:
    """Stop any running training process."""
    process = get_running_process()
    if process:
        process.stop()
