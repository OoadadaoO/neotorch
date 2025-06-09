package app.adada.neo4j.algo;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.Result;
import java.util.*;
import java.util.function.Predicate;
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
        List<Node> allNodes = new ArrayList<>(batchNodes);
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
                throw new IllegalStateException("No compliant neighbor found for node: " + n.getElementId());
            }
            Relationship chosenRel = compliantRels.get(random.nextInt(compliantRels.size()));
            Node posNode = chosenRel.getOtherNode(n);
            String posEid = posNode.getElementId();
            if (!nodeIndex.containsKey(posEid)) {
                nodeIndex.put(posEid, nodeIndex.size());
                allNodes.add(posNode);
            }
            positiveEdges[0][i] = nodeIndex.get(posEid);
            positiveEdges[1][i] = nodeIndex.get(n.getElementId());
        }

        // Negative sampling
        List<Node> pool = collectAllNodes(nodeLabels, tx, n -> true);
        if (pool.isEmpty()) {
            throw new IllegalStateException("No valid negative nodes found in the graph.");
        }
        for (int i = 0; i < batchNodes.size(); i++) {
            for (int j = 0; j < numNeg; j++) {
                Node negNode = pool.get(random.nextInt(pool.size()));
                String negEid = negNode.getElementId();
                if (!nodeIndex.containsKey(negEid)) {
                    nodeIndex.put(negEid, nodeIndex.size());
                    allNodes.add(negNode);
                }
                int negIndex = i * numNeg + j;
                negativeEdges[0][negIndex] = nodeIndex.get(negEid);
                negativeEdges[1][negIndex] = nodeIndex.get(batchNodes.get(i).getElementId());
            }
        }
        return new ExtendedBatch(allNodes, positiveEdges, negativeEdges);
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

    private List<Node> collectAllNodes(List<String> labels, Transaction tx, Predicate<Node> filter) {
        String cypher;
        if (labels.contains("*")) {
            cypher = "MATCH (n) RETURN n";
        } else {
            String lbls = labels.stream()
                    .map(l -> ":`" + l + "`")
                    .collect(Collectors.joining("|"));
            cypher = String.format("MATCH (n%s) RETURN n", lbls);
        }
        Result res = tx.execute(cypher);
        List<Node> all = new ArrayList<>();
        while (res.hasNext()) {
            Node node = (Node) res.next().get("n");
            if (filter == null || filter.test(node)) {
                all.add(node);
            }
        }
        return all;
    }
}
