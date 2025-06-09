import sys
import os
import time
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from neo4j import GraphDatabase
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neo4j_monitor import Neo4jMonitor


load_dotenv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWD")


def restart_neo4j():
    """Restart Neo4j service."""
    print("Restarting Neo4j...")
    os.system("sudo systemctl restart neo4j")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
    while True:
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            print("Waiting for it to stabilize...")
            break
        except Exception as e:
            time.sleep(1)
    time.sleep(5)  # Give Neo4j some time to stabilize after restart


def run_cypher_queries(queries: list[str] = None):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        for q in queries:
            q = q.strip()
            if not q:
                continue
            result = session.run(q)
            for record in result:
                print(record)


def test(query: list[str]):
    restart_neo4j()

    monitor = Neo4jMonitor()
    monitor.start()

    run_cypher_queries(query)

    monitor.stop()
    monitor.log_stats()

    return monitor.get_stats()


def plot_stats(stats: dict[dict], title: str, filename: str = "cora.png"):
    """Plot CPU and memory usage statistics."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for method, data in stats.items():
        ax1.plot(data["time"], data["cpu"], label=method)
        ax2.plot(data["time"], np.array(data["memory"]) / 10**9, label=method)
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
