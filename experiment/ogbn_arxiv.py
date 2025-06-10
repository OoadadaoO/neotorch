import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import plot_stats, test


GDS_QUERIES = """
MATCH (source:Paper)
OPTIONAL MATCH (source:Paper)-[r:CITES]->(target:Paper)
RETURN gds.graph.project(
  'ogbn-arxiv',
  source,
  target,
  {
    sourceNodeLabels: labels(source),
    targetNodeLabels: labels(target),
    sourceNodeProperties: source { .features },
    targetNodeProperties: target { .features },
    relationshipType: type(r)
  },
  { undirectedRelationshipTypes: ['CITES'] }
);
CALL gds.beta.graphSage.train(
  'ogbn-arxiv',
  {
    modelName: 'test-gds',
    featureProperties: ['features'],
    embeddingDimension: 256,
    aggregator: 'mean',
    activationFunction: 'relu',
    sampleSizes: [25, 10],
    batchSize: 512,
    learningRate: 0.001,
    epochs: 10,
    negativeSampleWeight: 1,
    maxIterations: 10,
    randomSeed: 42,
    tolerance: 0
  }
) YIELD modelInfo as info
RETURN
  info.modelName as modelName,
  info.metrics.didConverge as didConverge,
  info.metrics.ranEpochs as ranEpochs,
  info.metrics.epochLosses as epochLosses;
CALL gds.model.drop('test-gds');
CALL gds.graph.drop('ogbn-arxiv')
"""

NEOTORCH_QUERIES = """
MATCH (n)
WITH COLLECT(n) AS nodes
CALL neotorch.graphsage.train(
  "test-neotorch", 
  nodes, 
  {
    featureProperties: ['features'],
    featureDimension: 128,
    embeddingDimension: 256,
    aggregator: 'mean',
    activationFunction: 'relu',
    sampleSizes: [25, 10],
    batchSize: 512,
    learningRate: 0.001,
    epochs: 10,
    maxIterations: 10,
    randomSeed: 42
  }
) YIELD model
RETURN model;
CALL neotorch.graphsage.delete("test-neotorch")
"""


if __name__ == "__main__":
    gds_stats = test(GDS_QUERIES.split(";"))
    neotorch_stats = test(NEOTORCH_QUERIES.split(";"))

    plot_stats(
        {
            "GDS": gds_stats,
            "NeoTorch": neotorch_stats,
        },
        title="GDS vs NeoTorch Performance on ogbn-arxiv Dataset",
        filename=f"ogbn_arxiv_b_512_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.png",
    )
