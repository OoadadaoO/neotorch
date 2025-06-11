import os
import threading
import time
from dotenv import load_dotenv
from matplotlib import axes, pyplot as plt
from neo4j import GraphDatabase
import numpy as np
import psutil

load_dotenv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWD")


class Neo4jMonitor:
    def __init__(self):
        self.proc = self._find_neo4j_main_process()
        self.stop_event = threading.Event()
        self.log = {
            "time": [],
            "cpu": [],
            "memory": [],
        }
        self.stage = {
            "time": [],
            "cpu": [],
            "memory": [],
        }
        self.sub_stage = {
            "time": [],
            "cpu": [],
            "memory": [],
        }
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
            self.log["time"].append(current_ts - self.start_time)
            self.log["cpu"].append(current_cpu)
            self.log["memory"].append(current_memory)

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
                "log": self.log,
                "stage": self.stage,
                "sub_stage": self.sub_stage,
            }

    def log_stats(self):
        with self._lock:
            memory_peak_increase = max(self.log["memory"]) - self.log["memory"][0]
            cpu_peak = max(self.log["cpu"])
            cpu_avg = sum(self.log["cpu"]) / len(self.log["cpu"])
            print(
                f"=== Neo4j Performance Statistics ===\n"
                f"Execution time:       {self.end_time - self.start_time:.2f} seconds\n"
                f"Memory peak increase: {memory_peak_increase / (1024**2):.2f} MB\n"
                f"CPU peak usage:       {cpu_peak:.2f}%\n"
                f"CPU average usage:    {cpu_avg:.2f}%\n"
                f"===================================="
            )

    def add_stage(self):
        """Add a stage marker for plotting purposes"""
        self._update_stats()
        with self._lock:
            self.stage["time"].append(self.log["time"][-1])
            self.stage["cpu"].append(self.log["cpu"][-1])
            self.stage["memory"].append(self.log["memory"][-1])

    def add_sub_stage(self):
        """Add a sub-stage marker for plotting purposes"""
        self._update_stats()
        with self._lock:
            self.sub_stage["time"].append(self.log["time"][-1])
            self.sub_stage["cpu"].append(self.log["cpu"][-1])
            self.sub_stage["memory"].append(self.log["memory"][-1])


def restart_neo4j():
    """Restart Neo4j service."""
    print(">>> Restarting Neo4j...")
    os.system("sudo systemctl restart neo4j")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
    while True:
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            print(">>> Waiting for Neo4j to stabilize...")
            break
        except Exception as e:
            time.sleep(1)
    time.sleep(15)  # Give Neo4j some time to stabilize after restart


def run_cypher_queries(
    queries: list[str],
    hook: callable = lambda x: None,
    trigger: callable = lambda x: None,
):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        for i, q in enumerate(queries):
            q = q.strip()
            if not q:
                continue
            if i != 0:
                hook()
            result = session.run(q)
            for record in list(result):
                # Convert to list to force evaluation of the lazy query result
                if i == 1:
                    print(record)
                else:
                    trigger(record)


def test(query: list[str]):
    restart_neo4j()

    monitor = Neo4jMonitor()
    monitor.start()
    try:
        print(">>> Starting Neo4j performance monitoring...")
        time.sleep(5)

        for _ in range(2):
            print(">>> Running Cypher queries...")
            monitor.add_stage()
            run_cypher_queries(query, monitor.add_sub_stage, list)

            print(">>> Waiting for GC to complete...")
            monitor.add_stage()
            time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        monitor.stop()
        monitor.log_stats()
        return monitor.get_stats()


def plot_stats(stats: dict[dict], title: str, filename: str = "cora.png"):
    """Plot CPU and memory usage statistics."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    assert isinstance(ax1, plt.Axes) and isinstance(
        ax2, axes.Axes
    ), "Subplots must be Axes objects"
    for method, data in stats.items():
        ax1.plot(data["log"]["time"], data["log"]["cpu"], label=method)
        ax2.plot(
            data["log"]["time"], np.array(data["log"]["memory"]) / 10**9, label=method
        )
        ax1.plot(data["stage"]["time"], data["stage"]["cpu"], "k.", label="_nolegend_")
        ax2.plot(
            data["stage"]["time"],
            np.array(data["stage"]["memory"]) / 10**9,
            "k.",
            label="_nolegend_",
        )
        ax1.plot(
            data["sub_stage"]["time"],
            data["sub_stage"]["cpu"],
            "k,",
            label="_nolegend_",
        )
        ax2.plot(
            data["sub_stage"]["time"],
            np.array(data["sub_stage"]["memory"]) / 10**9,
            "k,",
            label="_nolegend_",
        )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("CPU Usage (%)")
    ax1.legend()
    ax1.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Memory Consumption (GB)")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(title)
    fig.tight_layout(h_pad=1.5)

    # Save the figure
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
