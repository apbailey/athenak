"""
Various utility functions used for automatic testing, including
  - functions for building code on target device (CPU/GPU)
  - functions for running code on target device
  - functions to clean up run directories at end of testing
"""

# Modules
import os
import shlex
from subprocess import Popen, PIPE
from typing import List
import time
import pytest
import logging
import sys

sys.path.insert(0, "../vis/python")
import athena_read  # noqa: E402

athena_read.check_nan_flag = True  # Enable NaN checking in athena_read

# Constants and configurations
ATHENAK_PATH = ".."
# Build directory holding the `athena` binary. Overridable via the environment so
# the SAME test code can target a prebuilt binary elsewhere (e.g. an arch-specific
# cluster build) with no edits and no rebuild:
#     export ATHENAK_BUILD=/home/u21/averybailey/athenak/build_p100/src   (absolute ok)
ATHENAK_BUILD = os.environ.get("ATHENAK_BUILD", "build/src")

# How to launch the binary is orthogonal to where it lives, so it is its own knob.
# Empty (the default) runs the binary directly (serial / single GPU); on a cluster:
#     export ATHENAK_LAUNCHER="mpirun -np 1"        # or "srun --mpi=pmix"
DEFAULT_LAUNCHER = shlex.split(os.environ.get("ATHENAK_LAUNCHER", ""))

# Configure logging
LOG_FILE_PATH = os.path.abspath(os.path.join(ATHENAK_PATH, "tst", "test_log.txt"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),  # Optional: Keep console logging
    ],
)


def run_command(command: List[str], text: bool = False) -> bool:
    """
    Executes a shell command and captures its output and errors.

    Args:
        command (list): The command to execute as a list of strings.
        text (bool): Whether to treat output and errors as text (default: False).

    Returns:
        bool: True if the command executed successfully, False otherwise.
    """

    logging.info(f"Executing command: {' '.join(command)}")
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
    # Log the output only to the file
    with open(LOG_FILE_PATH, "a") as log_file:
        output, errors = process.communicate()
        log_file.write(output)
        log_file.write(errors)

    if process.returncode != 0:
        logging.error(f"Command failed with return code {process.returncode}")
    return process.returncode == 0


def cmake(flags: List[str] = None, **kwargs) -> bool:
    """
    Runs the CMake command to configure the build system.

    Args:
        flags (list): Additional flags to pass to the CMake command.
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the CMake command succeeded, False otherwise.

    Raises:
        RuntimeError: If the CMake command fails.
    """
    if flags is None:
        flags = []

    original_dir = os.getcwd()
    try:
        os.chdir(ATHENAK_PATH)
        logging.info(f"Configuring CMake in {os.getcwd()}")

        command = ["cmake"] + flags + ["-B", "tst/build"]
        if not run_command(command, **kwargs):
            raise RuntimeError("CMake configuration failed")
    finally:
        os.chdir(original_dir)
    return True


def make(threads: int = os.cpu_count(), **kwargs) -> bool:
    """
    Runs the Make command to compile the project.

    Args:
        threads (int): number of threads to use for compilation (default: num of cores).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the Make command succeeded, False otherwise.

    Raises:
        RuntimeError: If the Make command fails.
    """
    os.chdir(ATHENAK_BUILD)
    command = ["make", "-j", f"{threads}"]
    start_time = time.time()
    status = run_command(command, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logging.info(f"make completed in {elapsed_time:.2f} seconds")
    if not status:
        raise RuntimeError("Make command failed")
    return True


def run(inputfile: str, flags=None, **kwargs) -> bool:
    """
    Executes a test case using the AthenaK binary.

    Args:
        inputfile (str): The path to the test case inputfile file.
        flags (list): Additional flags to pass to the AthenaK binary.
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the test case executed successfully, False otherwise.

    Raises:
        AssertionError: If the test case execution fails.

    Note:
        Prepends $ATHENAK_LAUNCHER (empty by default) so a cluster whose MPI build
        cannot run bare (e.g. OpenMPI that refuses MPI_Init without a launcher) can
        set e.g. ATHENAK_LAUNCHER="mpirun -np 1" without editing test code.
    """
    if flags is None:
        flags = []

    command = list(DEFAULT_LAUNCHER) + ["./athena", "-i", inputfile] + flags
    if not run_command(command, **kwargs):
        logging.error(f"Failed to execute {inputfile} with flags {flags}")
        raise RuntimeError(f"Failed to execute {inputfile} with flags {flags}")
    return True


def mpi_run(
    inputfile: str, flags=None, threads: int = min(16, os.cpu_count()), **kwargs
) -> bool:
    """
    Executes a test case using the AthenaK binary with MPI support.

    Args:
        inputfile (str): The path to the test case input file.
        flags (list): Additional flags to pass to the AthenaK binary.
        threads (int): Number of threads to use for MPI execution (default: smallest
            of (16) or (num of cores)).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the test case executed successfully, False otherwise.

    Raises:
        AssertionError: If the test case execution fails.
    """

    if flags is None:
        flags = []

    command = ["mpirun", "-np", str(threads), "./athena", "-i", inputfile] + flags
    if not run_command(command, **kwargs):
        logging.error(
           f"Failed to execute {inputfile} with flags {flags} using MPI "
           f"and {threads}-threads"
        )
        raise RuntimeError(
            f"Failed to execute {inputfile} with flags {flags} using MPI "
            f"and {threads}-threads"
        )
    return True


def run_command_capture(command: List[str], cwd: str = None):
    """
    Executes a shell command and RETURNS its output (unlike run_command, which
    only logs the output and returns a success bool).

    stdout/stderr are still appended to the shared log file for debugging, but
    are also returned so a caller can parse them (e.g. a throughput number that
    the binary prints to stdout).

    Args:
        command (list): The command to execute as a list of strings.
        cwd (str): Directory to run the command in (default: current directory).

    Returns:
        tuple: (returncode: int, stdout: str, stderr: str).
    """
    logging.info(f"Executing (capture): {' '.join(command)}")
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True, cwd=cwd)
    output, errors = process.communicate()
    # Log with a delimiter so per-run output is distinguishable in the shared log.
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"\n$ {' '.join(command)}\n")
        log_file.write(output)
        log_file.write(errors)
    if process.returncode != 0:
        logging.error(f"Command failed with return code {process.returncode}")
    return process.returncode, output, errors


def run_capture(
    inputfile: str,
    flags=None,
    *,
    launcher=None,
    cwd: str = None,
    check: bool = True,
) -> str:
    """
    Runs the AthenaK binary and RETURNS its captured stdout, so a test can parse
    printed values (throughput, timings, etc.). Mirrors run(), but returns output
    instead of a bool.

    The binary is `./athena` run from the build directory (ATHENAK_BUILD), the same
    convention run()/make() use; set the env var ATHENAK_BUILD to target a prebuilt
    binary elsewhere (e.g. an arch-specific cluster build).

    Args:
        inputfile (str): Path to the athinput file (relative to the run cwd).
        flags (list): Extra CLI overrides, e.g. ["mesh/nx1=64", "nr_radiation/sweep=diagonal"].
        launcher (list): Launch prefix, e.g. ["mpirun", "-np", "1"] or
            ["srun", "--mpi=pmix"]. Defaults to $ATHENAK_LAUNCHER (space-split) or none.
        cwd (str): Directory to run in. Defaults to ATHENAK_BUILD (where ./athena lives).
        check (bool): If True (default), raise RuntimeError on nonzero exit.

    Returns:
        str: Captured stdout.

    Raises:
        RuntimeError: If check is True and the run exits nonzero.
    """
    if flags is None:
        flags = []
    if launcher is None:
        launcher = DEFAULT_LAUNCHER
    if cwd is None:
        cwd = ATHENAK_BUILD

    command = list(launcher) + ["./athena", "-i", inputfile] + list(flags)
    returncode, output, errors = run_command_capture(command, cwd=cwd)
    if check and returncode != 0:
        logging.error(f"Failed to execute {inputfile} with flags {flags}")
        raise RuntimeError(
            f"Run failed (exit {returncode}) for {inputfile} with flags {flags}\n"
            f"--- stderr ---\n{errors}"
        )
    return output


def cleanup(text=False) -> None:
    """
    Cleans up the test environment by removing generated files.
    """
    if text:
        logging.info("Cleaning up test environment")
    Popen(["rm -rf tab/"], shell=True, stdout=PIPE).communicate()
    Popen(["rm " + "*.dat"], shell=True, stdout=PIPE).communicate()
    if text:
        logging.info("Cleanup completed")


def clean() -> None:
    """
    Cleans the build directory.
    """
    logging.info("Cleaning build directory")
    Popen(["rm -rf build/"], shell=True, stdout=PIPE).communicate()


def clean_make(threads: int = os.cpu_count(), **kwargs) -> None:
    """
    Cleans the build directory and rebuilds the project.
    Removes all files in the build directory and then runs CMake and Make.
    """
    clean()
    cmake(**kwargs)
    make(threads=threads)
    logging.info("Build directory cleaned and project rebuilt")
    run_command(["ln", "-s", "../../inputs", "inputs"])


def read_dictionary_from_file(file_path):
    """
    Reads a dictionary from a file where each line is in the format "key: value".

    Args:
        file_path: The path to the file.

    Returns:
        A dictionary, or None if an error occurred.
    """
    try:
        my_dict = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Skip empty lines
                    try:
                        keys, values = line.split(": ", 1)  # Split at the first ": "
                        values = values.strip("()")
                        error, ratio = values.split(",")
                        keys = keys.strip("()")
                        keys = keys.split(",")
                        keys = [key.strip(" '") for key in keys]
                        my_dict[tuple(keys)] = (float(error), float(ratio))
                    except ValueError:
                        print(f"Warning: Skipping invalid line: {line}")
        return my_dict
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None  # Or raise the exception, depending on desired behavior
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def test_error_convergence(
    input_file,
    test_name,
    arguments,
    errors,
    _wave,
    _res,
    iv,
    rv,
    fv,
    soe,
    left_wave="0",
    right_wave="0",
    mpi=False,
):
    RUN = mpi_run if mpi else run
    l1_rms_l = 0.0
    l1_rms_r = 0.0
    for wv in _wave:
        try:
            for res in _res:
                results = RUN(
                    input_file, arguments(iv, rv, fv, wv, res, soe, test_name)
                )
                assert results, f"Run failed for {soe}+{iv}+{res}+{fv}+{rv}+{wv}."
            maxerror, maxerrorratio = errors[(soe, iv, rv, wv)]
            data = athena_read.error_dat(f"{test_name}-errs.dat")
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_nLR = data[0][L1_RMS_INDEX]
            l1_rms_nHR = data[1][L1_RMS_INDEX]
            errorratio = l1_rms_nHR / l1_rms_nLR
            if l1_rms_nHR > maxerror and not (rv == "ppmx" and iv == "rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                pytest.fail(
                    f"{wv} wave error too large for {soe}+{iv}+{rv}+{fv},"
                    f"error: {l1_rms_nHR:g} threshold: {maxerror:g}"
                )
            if errorratio > maxerrorratio and not (rv == "ppmx" and iv == "rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                pytest.fail(
                    f"{wv} not converging for {soe}+{iv}+{rv}+{fv},"
                    f"error ratio: {errorratio:g} threshold: {maxerrorratio:g}"
                )
            # store errors for selected L/R-going waves
            if wv == left_wave:
                l1_rms_l = l1_rms_nHR
            if wv == right_wave:
                l1_rms_r = l1_rms_nHR
        finally:
            cleanup()
    return l1_rms_l, l1_rms_r
