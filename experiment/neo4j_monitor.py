import time
import threading
import psutil


class Neo4jMonitor:
    def __init__(self):
        self.proc = self._find_neo4j_main_process()
        self.stop_event = threading.Event()
        self.memory_peak_increase = 0
        self.cpu_peak = 0
        self.time_log = []
        self.cpu_log = []
        self.memory_log = []
        self.stage_points = {
            "time": [],
            "cpu": [],
            "memory": [],
        }
        self._memory_baseline = self.proc.memory_info().rss
        self._thread = threading.Thread(target=self._monitor_loop)
        self._lock = threading.Lock()

    def _find_neo4j_main_process(self):
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(proc.info["cmdline"])
                if "org.neo4j.server" in cmd and "-cp" in cmd and "plugins" in cmd:
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        raise RuntimeError("Cannot find Neo4j main process. Ensure Neo4j is running.")

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            self._update_stats()
            # time.sleep(0.1)

    def _update_stats(self):
        current_memory = self.proc.memory_info().rss
        current_cpu = self.proc.cpu_percent(interval=0.1)
        current_ts = time.perf_counter()

        with self._lock:
            delta_memory = current_memory - self._memory_baseline
            if delta_memory > self.memory_peak_increase:
                self.memory_peak_increase = delta_memory
            if current_cpu > self.cpu_peak:
                self.cpu_peak = current_cpu
            self.cpu_log.append(current_cpu)
            self.memory_log.append(current_memory)
            self.time_log.append(current_ts - self.start_time)

    def start(self):
        self.stop_event.clear()
        self._thread.start()
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        self.stop_event.set()
        self._thread.join()

    def get_stats(self):
        with self._lock:
            return {
                "time": self.time_log,
                "cpu": self.cpu_log,
                "memory": self.memory_log,
                "stage_points": self.stage_points,
            }

    def log_stats(self):
        with self._lock:
            print(
                f"=== Neo4j Performance Statistics ===\n"
                f"Execution time:       {self.end_time - self.start_time:.2f} seconds\n"
                f"Memory peak increase: {self.memory_peak_increase / (1024**2):.2f} MB\n"
                f"CPU peak usage:       {self.cpu_peak:.2f}%\n"
                f"CPU average usage:    {sum(self.cpu_log) / len(self.cpu_log):.2f}%\n"
                f"===================================="
            )

    def add_stage_point(self):
        """Add a stage marker for plotting purposes"""
        self._update_stats()
        with self._lock:
            self.stage_points["time"].append(self.time_log[-1])
            self.stage_points["cpu"].append(self.cpu_log[-1])
            self.stage_points["memory"].append(self.memory_log[-1])
