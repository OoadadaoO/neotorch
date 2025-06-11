package app.adada.neo4j.algo;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.ResourceIterable;
import org.neo4j.graphdb.Transaction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * BatchSampler for GraphSAGE unsupervised learning.
 */
public class BatchSampler {
    private final Transaction tx;
    private final Random random = new Random();

    public BatchSampler(Transaction tx) {
        this.tx = tx;
    }

    /**
     * Expend batch with positive and negative samples.
     * 
     * @param batchNodes        List of input nodes
     * @param nodeLabels        allowed node labels ("*" means any)
     * @param relationshipTypes allowed rel types ("*" means any)
     * @param numNeg            number of negatives per node
     * @return extended node list
     */
    public ExtendedBatch sampleBatch(List<Node> batchNodes,
            List<String> nodeLabels,
            List<String> relationshipTypes,
            int numNeg) {

        // Mapping node to index
        Map<String, Integer> nodeIndex = new HashMap<>((int) Math.ceil(batchNodes.size() * (2 + numNeg) / 0.75));
        List<Node> extendedNodes = new ArrayList<>(batchNodes);
        for (Node n : batchNodes)
            nodeIndex.put(n.getElementId(), nodeIndex.size());

        long[][] positiveEdges = new long[2][batchNodes.size()];
        long[][] negativeEdges = new long[2][batchNodes.size() * numNeg];

        for (int i = 0; i < batchNodes.size(); i++) {
            Node n = batchNodes.get(i);
            // Positive neighbors
            List<RelationshipType> relTypes = relationshipTypes.stream()
                    .map(RelationshipType::withName)
                    .collect(Collectors.toList());
            Iterable<Relationship> rels = (relationshipTypes.contains("*")
                    ? n.getRelationships(Direction.INCOMING)
                    : n.getRelationships(Direction.INCOMING, relTypes.toArray(new RelationshipType[0])));
            List<Relationship> compliantRels = StreamSupport.stream(rels.spliterator(), false)
                    .filter(r -> matchesLabels(r.getOtherNode(n), nodeLabels))
                    .collect(Collectors.toList());
            if (compliantRels.isEmpty()) {
                continue; // Skip nodes with no valid positive neighbors
            }
            Relationship chosenRel = compliantRels.get(random.nextInt(compliantRels.size()));
            Node posNode = chosenRel.getOtherNode(n);
            String posEid = posNode.getElementId();
            if (!nodeIndex.containsKey(posEid)) {
                nodeIndex.put(posEid, nodeIndex.size());
                extendedNodes.add(posNode);
            }
            positiveEdges[0][i] = nodeIndex.get(posEid);
            positiveEdges[1][i] = nodeIndex.get(n.getElementId());
        }

        // Negative sampling
        try (ResourceIterable<Node> allNodes = tx.getAllNodes()) {
            // List<Node> selectedNodes = ReservoirSampling.sample(allNodes.iterator(),
            // batchNodes.size() * numNeg,
            // (node) -> matchesLabels(node, nodeLabels) &&
            // !nodeIndex.containsKey(node.getElementId()));
            // if (selectedNodes.isEmpty()) {
            // throw new IllegalStateException("No valid negative nodes found in the
            // graph.");
            // }
            // for (int i = 0; i < selectedNodes.size(); i++) {
            // Node negNode = selectedNodes.get(i);
            // String negEid = negNode.getElementId();
            // nodeIndex.put(negEid, nodeIndex.size());
            // extendedNodes.add(negNode);
            // int batchIndex = i / numNeg;
            // negativeEdges[0][i] = nodeIndex.get(negEid);
            // negativeEdges[1][i] =
            // nodeIndex.get(batchNodes.get(batchIndex).getElementId());
            // }
            List<Node> pool = allNodes.stream().toList();
            if (pool.isEmpty()) {
                throw new IllegalStateException("No valid negative nodes found in the graph.");
            }
            for (int i = 0; i < batchNodes.size(); i++) {
                for (int j = 0; j < numNeg; j++) {
                    Node negNode = pool.get(random.nextInt(pool.size()));
                    String negEid = negNode.getElementId();
                    if (!nodeIndex.containsKey(negEid)) {
                        nodeIndex.put(negEid, nodeIndex.size());
                        extendedNodes.add(negNode);
                    }
                    int negIndex = i * numNeg + j;
                    negativeEdges[0][negIndex] = nodeIndex.get(negEid);
                    negativeEdges[1][negIndex] = nodeIndex.get(batchNodes.get(i).getElementId());
                }
            }
        }
        return new ExtendedBatch(extendedNodes, positiveEdges, negativeEdges);
    }

    public class ExtendedBatch {
        public final List<Node> nodes;
        public final long[][] positiveEdges;
        public final long[][] negativeEdges;

        public ExtendedBatch(List<Node> nodes, long[][] positiveEdges, long[][] negativeEdges) {
            this.nodes = nodes;
            this.positiveEdges = positiveEdges;
            this.negativeEdges = negativeEdges;
        }
    }

    private boolean matchesLabels(Node n, List<String> labels) {
        if (labels.contains("*"))
            return true;
        for (String lbl : labels) {
            if (n.hasLabel(() -> lbl))
                return true;
        }
        return false;
    }
}
