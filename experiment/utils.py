import sys
import os
import time
from dotenv import load_dotenv
from matplotlib import axes, lines, markers, pyplot as plt
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
    trigger: callable = lambda x: None,
):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        for q in queries:
            q = q.strip()
            if not q:
                continue
            result = session.run(q)
            for record in list(
                result
            ):  # Convert to list to force evaluation of the lazy query result
                trigger(record)


def test(query: list[str]):
    restart_neo4j()

    monitor = Neo4jMonitor()
    monitor.start()

    print(">>> Starting Neo4j performance monitoring...")
    time.sleep(5)

    # print(">>> Warming up page cache...")
    # monitor.add_stage_point()
    # warmup_queries = [
    #     """
    #     MATCH (n)
    #     OPTIONAL MATCH (n)-[r]->()
    #     RETURN count(n.features) + count(r);
    #     """,
    # ]
    # run_cypher_queries(warmup_queries, print)

    for _ in range(2):
        print(">>> Running Cypher queries...")
        monitor.add_stage_point()
        run_cypher_queries(query, print)

        print(">>> Waiting for GC to complete...")
        monitor.add_stage_point()
        time.sleep(5)

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
        ax1.plot(data["time"], data["cpu"], label=method)
        ax1.plot(
            data["stage_points"]["time"],
            data["stage_points"]["cpu"],
            linestyle="None",
            marker=".",
            color="black",
            label="_nolegend_",
        )
        ax2.plot(data["time"], np.array(data["memory"]) / 10**9, label=method)
        ax2.plot(
            data["stage_points"]["time"],
            np.array(data["stage_points"]["memory"]) / 10**9,
            linestyle="None",
            marker=".",
            color="black",
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
