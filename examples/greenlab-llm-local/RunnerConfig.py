
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.OperationType import OperationType
import shutil
import shlex
import csv
import statistics


class RunnerConfig:
    name = "greenlab_gpu_smoke_v1"
    results_output_path =Path.home() / "GreenLab" / "results" #"/home/mahesh/Desktop/courses/Green_lab/lab_repo/experiment-runner/examples/greenlab-llm/results" #Path.home() / "GreenLab" / "results"
    operation_type = OperationType.AUTO
    time_between_runs_in_ms = 5000

    LLAMA = Path(
        "/media/mahesh/New Volume/courses/"
        "GreenLab/repo/llama.cpp/build-cuda/bin/llama-server"
    )

    # ENERGIBRIDGE = Path.home() / (
    #     "EnergiBridge/target/release/energibridge"
    # )

    ENERGIBRIDGE = shutil.which("energibridge")

    MODELS = {
        "gemma4b": "ggml-org/gemma-3-4b-it-GGUF",
        "qwen3b": (
            "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
        ),
        "qwen7b": (
            "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"
        ),
    }

    GPU_LAYERS = {
        "gemma4b": 999,
        "qwen3b": 999,
        "qwen7b": 20,
    }

    URL = "http://127.0.0.1:8080"

    PROMPT = (
        "Explain photosynthesis, including the roles of sunlight, "
        "water, carbon dioxide, chlorophyll, and glucose. "
        "Write approximately {word_count} words in clear, "
        "continuous prose."
    )
    

    def __init__(self):
        self.server = None
        self.server_log = None
        self.current_model = None
        self.metrics = {}

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
        ])

    def calculate_resource_metrics(self, csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError("EnergiBridge CSV is empty")

        # CPU usage: average across logical CPUs and samples
        cpu_columns = [
            col for col in rows[0]
            if col.startswith("CPU_USAGE_")
        ]

        if not cpu_columns:
            raise ValueError("CPU usage columns not found")

        cpu_values = [
            statistics.mean(float(row[col]) for col in cpu_columns)
            for row in rows
        ]

        mean_cpu = statistics.mean(cpu_values)

        # System memory usage in MiB
        memory_values = [
            float(row["USED_MEMORY"]) / (1024 ** 2)
            for row in rows
        ]

        mean_memory = statistics.mean(memory_values)

        # CPU package energy in joules
        energy_col = next(
            (
                col for col in [
                    "PACKAGE_ENERGY (J)",
                    "CPU_ENERGY (J)"
                ]
                if col in rows[0]
            ),
            None
        )

        if energy_col is None:
            raise ValueError(
                "No supported cumulative energy column found"
            )

        energy_start = float(rows[0][energy_col])
        energy_end = float(rows[-1][energy_col])

        total_energy = energy_end - energy_start

        if total_energy < 0:
            raise ValueError(
                "Energy counter decreased; check counter wrap/reset"
            )

        return {
            "mean_cpu_usage": round(mean_cpu, 2),
            "mean_memory_usage": round(mean_memory, 2),
            "mean_energy_usage": round(total_energy, 4),
        }


    def create_run_table_model(self):
        self.run_table_model = RunTableModel(
            factors=[
                FactorModel("model", list(self.MODELS.keys())),
                FactorModel("max_tokens", [1000])
            ],
            repetitions=2,
            data_columns=[
                "duration_s",
                "prompt_tokens",
                "completion_tokens",
                "output_bytes",
                "tokens_per_second",
                    "mean_cpu_usage",
                    "mean_memory_usage",
                    "mean_energy_usage"
            ],
            shuffle=False
        )
        return self.run_table_model

    def before_experiment(self):
        self.results_output_path.mkdir(
            parents=True, exist_ok=True
        )

        if not self.LLAMA.is_file():
            raise FileNotFoundError(self.LLAMA)

        if not self.ENERGIBRIDGE:
            # raise FileNotFoundError(self.ENERGIBRIDGE)
            raise FileNotFoundError("EnergiBridge executable not found in PATH")

    def before_run(self):
        pass

    def server_ready(self):
        try:
            with urllib.request.urlopen(
                self.URL + "/health", timeout=3
            ) as response:
                data = json.load(response)
                return data.get("status") == "ok"
        except Exception:
            return False

    def stop_server(self):
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
            self.server = None

        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None

        self.current_model = None

    def start_run(self, context):
        model = context.execute_run["model"]

        # Reuse the loaded server for consecutive runs
        # of the same model.
        if self.current_model == model and self.server_ready():
            return

        self.stop_server()

        log_path = (
            self.results_output_path
            / self.name
            / f"server_{model}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.server_log = open(log_path, "a")

        command = [
            str(self.LLAMA),
            "-hf", self.MODELS[model],
            "--host", "127.0.0.1",
            "--port", "8080",
            "-c", "2048",
            "-t", "8",
            "-np", "1",
            "-ngl", str(self.GPU_LAYERS[model]),
        ]

        self.server = subprocess.Popen(
            command,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
        )
        self.current_model = model

        deadline = time.monotonic() + 300

        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited: {log_path}"
                )
            if self.server_ready():
                break
            time.sleep(2)
        else:
            raise TimeoutError(
                f"Model did not become ready: {log_path}"
            )

        # Warm-up excluded from measured run.
        self.send_request(
            self.PROMPT, 16,
            context.run_dir / "warmup.json"
        )

    def send_request(self, prompt, tokens, output_path):
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": int(tokens),
            "temperature": 0,
            "stream": False,
        }

        payload_path = output_path.with_suffix(".request.json")
        payload_path.write_text(json.dumps(payload))

        return payload_path

    def start_measurement(self, context):
        # EnergiBridge is started in interact() so it
        # wraps precisely the HTTP request.
        pass

    def interact(self, context):
        tokens = int(context.execute_run["max_tokens"])
        run_dir = context.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        response_path = run_dir / "response.json"
        energy_path = run_dir / "energibridge.csv"

        payload_path = self.send_request(
            self.PROMPT, tokens,
            response_path
        )

        # command = [
        #     str(self.ENERGIBRIDGE),
        #     "--gpu",
        #     "--summary",
        #     "-o", str(energy_path),
        #     "--",
        #     "curl",
        #     "--fail-with-body",
        #     "--silent",
        #     "--show-error",
        #     "--max-time", "180",
        #     "-o", str(response_path),
        #     "-H", "Content-Type: application/json",
        #     "--data-binary", "@" + str(payload_path),
        #     self.URL + "/v1/chat/completions",
        # ]

        # HTTP inference request

        curl_command = [
            "curl",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--max-time", "180",
            "-o", str(response_path),
            "-H", "Content-Type: application/json",
            "--data-binary", "@" + str(payload_path),
            self.URL + "/v1/chat/completions",
        ]

        # EnergiBridge wraps the HTTP request.
        energy_command = [
            str(self.ENERGIBRIDGE),
            "--gpu",
            "--summary",
            "-o", str(energy_path),
            "--",
            *curl_command,
        ]

        # Run EnergiBridge using the msr group.
        command = [
            "sudo",
            "-n",
            "sg",
            "msr",
            "-c",
            shlex.join(energy_command),
        ]

        start = time.perf_counter()

        subprocess.run(
            command,
            check=True,
            timeout=210,
        )

        elapsed = time.perf_counter() - start

        # start = time.perf_counter()
        # subprocess.run(command, check=True, timeout=210)
        # elapsed = time.perf_counter() - start

        response = json.loads(response_path.read_text())
        usage = response.get("usage", {})

        output_text = response["choices"][0]["message"]["content"]
        generated = int(usage.get("completion_tokens", 0))

        self.metrics = {
            "duration_s": round(elapsed, 4),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": generated,
            "output_bytes": len(output_text.encode("utf-8")),
            "tokens_per_second": (
                round(generated / elapsed, 4)
                if elapsed > 0 else 0
            ),
        }

        resource_metrics = self.calculate_resource_metrics( energy_path)

        self.metrics.update(resource_metrics)

        (run_dir / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2)
        )

    def stop_measurement(self, context):
        pass

    def stop_run(self, context):
        pass

    def populate_run_data(self, context):
        return self.metrics

    def after_experiment(self):
        self.stop_server()

    experiment_path = None
