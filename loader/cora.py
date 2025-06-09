import csv
import os
from tqdm import tqdm
import subprocess
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Constants
DATASET = "cora"
NEO4J_HOME = "/var/lib/neo4j"
ARRAY_DELIMITER = ","
DELIMITER = ";"
QUOTE = "'"
MONO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
NODE_FILE = f"{DATASET}-nodes.csv"
EDGE_FILE = f"{DATASET}-edges.csv"

os.makedirs(os.path.join(MONO_ROOT, "import"), exist_ok=True)
output_node = os.path.join(MONO_ROOT, "import", NODE_FILE)
output_edge = os.path.join(MONO_ROOT, "import", EDGE_FILE)
neo4j_node = os.path.join(NEO4J_HOME, "import", NODE_FILE)
neo4j_edge = os.path.join(NEO4J_HOME, "import", EDGE_FILE)

NODE_LABEL = "Paper"
EDGE_TYPE = "CITES"

# Load dataset
root = os.path.join(MONO_ROOT, "data", "planetoid")
dataset = Planetoid(root=root, name=DATASET, transform=NormalizeFeatures())
graph = dataset[0]
print(f"Dataset: {graph}")
labels = graph["y"].numpy()
train_idx, valid_idx, test_idx = (
    graph["train_mask"],
    graph["val_mask"],
    graph["test_mask"],
)

# Stream write nodes
if not os.path.exists(output_node):
    with open(output_node, "w", newline="", encoding="utf-8") as node_file:
        node_writer = csv.writer(
            node_file, delimiter=DELIMITER, quotechar=QUOTE, quoting=csv.QUOTE_MINIMAL
        )

        # Header
        node_writer.writerow(
            ["id:ID{id-type:long}", "label:int", "features:float[]", ":LABEL"]
        )

        # data
        for i, (feat, label) in tqdm(
            enumerate(zip(graph["x"], labels)), total=len(labels)
        ):
            features = ARRAY_DELIMITER.join(map(str, feat.numpy()))
            split = (
                "TRAIN"
                if train_idx[i]
                else "VALID" if valid_idx[i] else "TEST" if test_idx[i] else "UNKNOWN"
            )
            node_writer.writerow(
                [i, int(label), features, ARRAY_DELIMITER.join([NODE_LABEL, split])]
            )

# Stream write edges
if not os.path.exists(output_edge):
    with open(output_edge, "w", newline="", encoding="utf-8") as edge_file:
        edge_writer = csv.writer(
            edge_file, delimiter=DELIMITER, quotechar=QUOTE, quoting=csv.QUOTE_MINIMAL
        )

        # Header
        edge_writer.writerow([":START_ID", ":END_ID"])

        # data
        src_list, dst_list = graph["edge_index"]
        # for src, dst in tqdm(
        #     np.unique(graph["edge_index"].T, axis=0), total=len(graph["edge_index"][0])
        # ):
        for src, dst in tqdm(
            zip(src_list.numpy(), dst_list.numpy()), total=len(src_list)
        ):
            edge_writer.writerow([src, dst])

# Import into Neo4j
print("Stopping Neo4j service to import data...")
subprocess.run(["sudo", "systemctl", "stop", "neo4j"], check=True)

print("Copy files to Neo4j import directory...")
subprocess.run(["sudo", "cp", output_node, neo4j_node], check=True)
subprocess.run(["sudo", "cp", output_edge, neo4j_edge], check=True)

print("Importing data into Neo4j...")
subprocess.run(
    [
        "sudo",
        "-u",
        "neo4j",
        "neo4j-admin",
        "database",
        "import",
        "full",
        "neo4j",
        "--overwrite-destination",
        f"--report-file={NEO4J_HOME}/import/import.report",
        f"--delimiter={DELIMITER}",
        f"--array-delimiter={ARRAY_DELIMITER}",
        f"--quote={QUOTE}",
        f"--nodes={neo4j_node}",
        f"--relationships={EDGE_TYPE}={neo4j_edge}",
    ],
    check=True,
)

print("Data import completed. Restarting Neo4j service...")
subprocess.run(["sudo", "systemctl", "start", "neo4j"], check=True)
