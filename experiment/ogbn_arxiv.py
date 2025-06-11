import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import plot_stats, test

GDS_PROJECT_GRAPH = """
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
)
"""
GDS_TRAIN_MODEL = """
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
    epochs: 5,
    negativeSampleWeight: 1,
    maxIterations: 10,
    randomSeed: 42,
    tolerance: 0
  }
) YIELD modelInfo as info, configuration
RETURN
  info.modelName as modelName,
  info.metrics.ranEpochs as ranEpochs,
  info.metrics.epochLosses as epochLosses,
  configuration
"""
GDS_DROP_MODEL = "CALL gds.model.drop('test-gds')"
GDS_DROP_GRAPH = "CALL gds.graph.drop('ogbn-arxiv')"

NEOTORCH_TRAIN_MODEL = """
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
    maxIterations: 5,
    randomSeed: 42
  }
) YIELD modelInfo as info, configuration
RETURN
  info.modelName as modelName,
  info.metrics.ranEpochs as ranEpochs,
  info.metrics.epochLosses as epochLosses,
  configuration
"""
NEOTORCH_DROP_MODEL = 'CALL neotorch.graphsage.drop("test-neotorch")'

if __name__ == "__main__":
    neotorch_stats = test(
        [NEOTORCH_DROP_MODEL, NEOTORCH_TRAIN_MODEL, NEOTORCH_DROP_MODEL]
    )
    gds_stats = test(
        [GDS_PROJECT_GRAPH, GDS_TRAIN_MODEL, GDS_DROP_MODEL, GDS_DROP_GRAPH]
    )

    plot_stats(
        {
            "GDS": gds_stats,
            "NeoTorch": neotorch_stats,
        },
        title="GDS vs NeoTorch Performance on ogbn-arxiv Dataset",
        filename=f"ogbn_arxiv_b_512_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.png",
    )
