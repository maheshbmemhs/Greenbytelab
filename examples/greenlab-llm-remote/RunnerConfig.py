from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import time
import requests
import base64
import subprocess

class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "new_runner_experiment_10"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path            = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 10000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    REMOTE_LLMS = {
            "qwen2.5-vl-3b": "http://127.0.0.1:8081/v1/chat/completions",
            "qwen2.5-vl-7b": "http://127.0.0.1:8082/v1/chat/completions",
            "gemma-3-4b":    "http://127.0.0.1:8083/v1/chat/completions",
    }

    NETWORK_INTERFACE = "eth0"

    IMAGE_PATH = Path("/home/kali/Downloads/flower.jpeg")

    NETWORK_CONDITIONS = {
        "normal":   {"delay": None,    "rate": None},
        "degraded": {"delay": "50ms", "rate": "20mbit"},
        "bad": {"delay": "100ms", "rate": "5mbit"},
    }

    CONTENT_SIZES = {
        "short":  {"max_tokens": 120,  "text_prompt": "Give a brief definition of football in a short paragraph with at least 50 words.",
                                        "image_prompt": "Briefly describe what is shown in this image with at least 50 words."},
        "medium": {"max_tokens": 650,  "text_prompt": "Write a paragraph about the history and rules of football since beginning up until now with at least 300 words.",
                                        "image_prompt": "Describe this image in a few sentences, tell me everything you know about it with at least 300 words."},
        "long":   {"max_tokens": 1350, "text_prompt": "Write a detailed essay about the evolution, rules, and cultural impact of football with at least 800 words and give examples of different teams.",
                                        "image_prompt": "Describe this image in extensive detail, covering all visible objects, colors, composition, and setting with at least 800 words."},
    }

    CONTENT_TYPES = ["text", "image"]

    REMOTE_HOST = "gl_greenbyte@glg3"
    LOCAL_ENERGIBRIDGE_CMD = ["sudo", "energibridge"]
    REMOTE_LOG_DIR         = "logs"
    STOP_FLAG_NAME         = "local_stop_flag"
    
    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later

        output.console_log("Custom config loaded")
             
    def create_run_table_model(self) -> RunTableModel:
        factor_model   = FactorModel("model", ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "gemma-3-4b"])
        factor_type    = FactorModel("content_type", ["text", "image"])
        factor_size    = FactorModel("content_size", ["short", "medium", "long"])
        factor_network = FactorModel("network_condition", ["normal", "degraded", "bad"])

        self.run_table_model = RunTableModel(
            factors=[factor_model, factor_type, factor_size, factor_network],
            exclude_combinations=[],
            repetitions=30,
            shuffle=True,
            data_columns=['max_tokens', 'duration', 'tokens_generated', 'word_count', 'output_bytes', 'tokens_per_second',
                          'start_ts', 'end_ts']
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("Config.before_experiment() called!")
        self._start_energibridge()

    def before_run(self) -> None:
        output.console_log("Config.before_run() called!")

    def _set_network_condition(self, net_label: str) -> None:
        cond = self.NETWORK_CONDITIONS[net_label]
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", self.NETWORK_INTERFACE, "root"],
                        stderr=subprocess.DEVNULL)
        if cond["delay"] or cond["rate"]:
            cmd = ["sudo", "tc", "qdisc", "add", "dev", self.NETWORK_INTERFACE, "root", "netem"]
            if cond["delay"]:
                cmd += ["delay", cond["delay"]]
            if cond["rate"]:
                cmd += ["rate", cond["rate"]]
            subprocess.run(cmd, check=True)

    def _reset_network_condition(self) -> None:
        subprocess.run(["sudo", "tc", "qdisc", "del", "dev", self.NETWORK_INTERFACE, "root"],
                        stderr=subprocess.DEVNULL)

    def start_run(self, context: RunnerContext) -> None:
        net_label = context.execute_run['network_condition']
        output.console_log(f"Starting run: {context.execute_run}")
        self._set_network_condition(net_label)
        time.sleep(1)

    def start_measurement(self, context: RunnerContext) -> None:
        pass # the energibridge start function added at the end

    def interact(self, context: RunnerContext) -> None:
        model        = context.execute_run['model']
        content_type = context.execute_run['content_type']
        content_size = context.execute_run['content_size']

        endpoint = self.REMOTE_LLMS[model]
        size_cfg = self.CONTENT_SIZES[content_size]

        if content_type == "text":
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": size_cfg["text_prompt"]}],
                "max_tokens": size_cfg["max_tokens"],
                "temperature": 0
            }
        else:
            with open(self.IMAGE_PATH, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": size_cfg["image_prompt"]},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                "max_tokens": size_cfg["max_tokens"],
                "temperature": 0
            }

        t0 = time.time()
        response = requests.post(endpoint, json=payload, timeout=180)
        t1 = time.time()

        self._last_response = response.json()
        self._request_start = t0
        self._request_end = t1
        output.console_log(f"Request took {t1 - t0:.2f}s")

    def stop_measurement(self, context: RunnerContext) -> None:
        pass # the energibridge stop function added at the end

    def stop_run(self, context: RunnerContext) -> None:
        self._reset_network_condition()

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        content_text = self._last_response["choices"][0]["message"]["content"]
        word_count = len(content_text.split())
        tokens = self._last_response.get("usage", {}).get("completion_tokens", None)
        response_time_s = round(self._request_end - self._request_start, 3)

        return {
            "max_tokens": self.CONTENT_SIZES[context.execute_run['content_size']]["max_tokens"],
            "duration": response_time_s,
            "tokens_generated": tokens,
            "word_count": word_count,
            "output_bytes": len(content_text.encode("utf-8")),
+           "tokens_per_second": round(tokens / response_time_s, 2) if tokens and response_time_s else None,
            "start_ts": self._request_start,
            "end_ts": self._request_end,
        }

    def after_experiment(self) -> None:
        self._reset_network_condition()
        self._stop_energibridge_and_fetch()
        output.console_log("Experiment complete")

    def _start_energibridge(self) -> None:
        local_csv  = self.experiment_path / "local.csv"
        local_flag = self.experiment_path / "local_stop_flag"
        local_flag.unlink(missing_ok=True)
        remote_csv = f"{self.REMOTE_LOG_DIR}/{self.name}_remote.csv"
        remote_flag = f"{self.REMOTE_LOG_DIR}/{self.STOP_FLAG_NAME}"

        subprocess.Popen(
            self.LOCAL_ENERGIBRIDGE_CMD + ["--output", str(local_csv), "--",
                "bash", "-c", f"while [ ! -f '{local_flag}' ]; do sleep 0.5; done"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

        remote_cmd = (
            f"mkdir -p {self.REMOTE_LOG_DIR} && rm -f {remote_flag}; "
            f"nohup energibridge --output {remote_csv} -- "
            f"bash -c 'while [ ! -f {remote_flag} ]; do sleep 0.5; done' "
            f"> /dev/null 2>&1 < /dev/null &"
        )
        subprocess.run(["ssh", self.REMOTE_HOST, remote_cmd], check=True)
        time.sleep(2)

    def _stop_energibridge_and_fetch(self) -> None:
        local_flag = self.experiment_path / "local_stop_flag"
        remote_csv = f"{self.REMOTE_LOG_DIR}/{self.name}_remote.csv"
        remote_flag = f"{self.REMOTE_LOG_DIR}/{self.STOP_FLAG_NAME}"
        local_flag.touch()
        subprocess.run(["ssh", self.REMOTE_HOST, f"touch {remote_flag}"], check=True)
        time.sleep(3)
        subprocess.run(["scp","-O", f"{self.REMOTE_HOST}:{remote_csv}",
                        str(self.experiment_path / "remote_energy.csv")], check=True)

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None