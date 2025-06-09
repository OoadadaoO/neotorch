package app.adada.neo4j.algo;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Multi-hop neighbor sampler for GraphSAGE.
 */
public class NeighborSampler {
    private final Random random = new Random();

    public NeighborSampler() {
    }

    /**
     * Sample neighbors for mini-batch training.
     * 
     * @param batchNodes        initial batch
     * @param featureProperties node float properties
     * @param nodeLabels        filter labels
     * @param relationshipTypes filter rel types
     * @param sampleSizes       sizes per hop (reverse order)
     * @return a Pair of features and edge indices
     */
    public DenseGraph sample(List<Node> batchNodes,
            List<String> featureProperties,
            List<String> nodeLabels,
            List<String> relationshipTypes,
            List<Long> immutableSampleSizes) {
        // Reverse sampleSizes for layer order
        List<Long> sampleSizes = new ArrayList<>(immutableSampleSizes);
        Collections.reverse(sampleSizes);

        // Mapping node to index
        Map<String, Integer> nodeIndex = new HashMap<>();
        List<Node> allNodes = new ArrayList<>(batchNodes);
        for (Node n : batchNodes)
            nodeIndex.put(n.getElementId(), nodeIndex.size());

        List<long[]> edges = new ArrayList<>();
        List<Node> frontier = new ArrayList<>(batchNodes);
        for (int hop = 0; hop < sampleSizes.size(); hop++) {
            int numSample = sampleSizes.get(hop).intValue();
            List<Node> nextFrontier = new ArrayList<>();
            for (Node n : frontier) {
                // incoming neighbors
                List<RelationshipType> relTypes = relationshipTypes.stream()
                        .map(RelationshipType::withName)
                        .collect(Collectors.toList());
                Iterable<Relationship> rels = (relationshipTypes.contains("*")
                        ? n.getRelationships(Direction.INCOMING)
                        : n.getRelationships(Direction.INCOMING, relTypes.toArray(new RelationshipType[0])));

                List<Relationship> compliantRels = StreamSupport.stream(rels.spliterator(), false)
                        .filter(r -> matchesLabels(r.getOtherNode(n), nodeLabels))
                        .collect(Collectors.toList());
                List<Node> reservior = new ArrayList<>();
                for (int i = 0; i < compliantRels.size(); i++) {
                    Relationship r = compliantRels.get(i);
                    Node neigh = r.getOtherNode(n);

                    if (i < numSample) {
                        reservior.add(neigh);
                    } else {
                        int idx = random.nextInt(i + 1);
                        if (idx < numSample) {
                            reservior.set(idx, neigh);
                        }
                    }
                }
                for (Node neigh : reservior) {
                    String eid = neigh.getElementId();
                    if (!nodeIndex.containsKey(eid)) {
                        // If the neighbor node is not in the index, add it
                        nodeIndex.put(eid, nodeIndex.size());
                        allNodes.add(neigh);
                        nextFrontier.add(neigh);
                    }
                    int src = nodeIndex.get(eid);
                    int dst = nodeIndex.get(n.getElementId());
                    edges.add(new long[] { src, dst });
                }
            }
            frontier = nextFrontier;

        }
        // Build feature matrix
        int N = allNodes.size();
        int D = countFeatureDim(featureProperties, allNodes.get(0));
        float[][] x = new float[N][D];
        for (int i = 0; i < N; i++) {
            x[i] = concatFeatures(allNodes.get(i), featureProperties);
        }
        // Build edge index array
        long[][] edgeIndex = new long[2][edges.size()];
        for (int i = 0; i < edges.size(); i++) {
            edgeIndex[0][i] = edges.get(i)[0];
            edgeIndex[1][i] = edges.get(i)[1];
        }
        return new DenseGraph(x, edgeIndex);
    }

    // private String buildQuery(List<String> nodeLabels,
    // List<String> relationshipTypes,
    // int limit) {
    // String labelFilter = nodeLabels.contains("*") ? ""
    // : nodeLabels.stream().map(l -> ":`" + l + "`")
    // .collect(Collectors.joining("|"));
    // String relFilter = relationshipTypes.contains("*") ? ""
    // : relationshipTypes.stream().map(r -> "`:" + r + "`")
    // .collect(Collectors.joining("|"));
    // return String.format("MATCH (m%s)-[r%s]->(n) WHERE n.elementId = $eid RETURN
    // m LIMIT %d",
    // labelFilter, relFilter, limit);
    // }

    private boolean matchesLabels(Node n, List<String> labels) {
        if (labels.contains("*"))
            return true;
        for (String lbl : labels) {
            if (n.hasLabel(() -> lbl))
                return true;
        }
        return false;
    }

    private int countFeatureDim(List<String> featureProperties, Node any) {
        // Assume first in db
        int dim = 0;
        for (String prop : featureProperties) {
            Object val = any.getProperty(prop);
            if (val instanceof Number)
                dim += 1;
            else if (val instanceof double[])
                dim += ((double[]) val).length;
            else if (val instanceof float[])
                dim += ((float[]) val).length;
            // add others as needed
        }
        return dim;
    }

    private float[] concatFeatures(Node n, List<String> featureProperties) {
        List<Float> feats = new ArrayList<>();
        for (String prop : featureProperties) {
            Object val = n.getProperty(prop);
            if (val instanceof Number)
                feats.add(((Number) val).floatValue());
            else if (val instanceof double[]) {
                for (double d : (double[]) val)
                    feats.add((float) d);
            } else if (val instanceof float[]) {
                for (float f : (float[]) val)
                    feats.add(f);
            }
        }
        float[] arr = new float[feats.size()];
        for (int i = 0; i < feats.size(); i++)
            arr[i] = feats.get(i);
        return arr;
    }
}