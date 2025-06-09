package app.adada.neo4j.procedure;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.Transaction;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import static app.adada.neo4j.util.ReservoirSampling.reservoirSample;;

public class NeighborSample {

    @Context
    public Transaction tx;

    @Procedure(name = "torch.graphsageSample", mode = Mode.READ)
    @Description("CALL torch.graphsageSample(label, source, relationshipType, neighborNumbers) YIELD sampledNodes")
    public Stream<SampledNodesResult> graphsageSample(
            @Name("label") String labelName,
            @Name("source") Node sourceNode,
            @Name("relationshipType") String relTypeName,
            @Name("neighborNumbers") List<Long> neighborNumbers) {

        Label label = Label.label(labelName);
        RelationshipType relationshipType = RelationshipType.withName(relTypeName);

        List<List<Node>> layers = new ArrayList<>();
        Set<String> visitedIds = new HashSet<>();

        layers.add(Collections.singletonList(sourceNode));
        visitedIds.add(sourceNode.getElementId());

        for (int depth = 0; depth < neighborNumbers.size(); depth++) {
            List<Node> previousLayer = layers.get(depth);
            Set<Node> currentLayerSet = new HashSet<>();

            for (Node node : previousLayer) {
                Iterable<Relationship> rels = node.getRelationships(Direction.OUTGOING, relationshipType);
                List<Node> neighbors = reservoirSample(
                        StreamSupport.stream(rels.spliterator(), false)
                                .map(r -> r.getOtherNode(node))
                                .filter(n -> n.hasLabel(label))
                                .iterator(),
                        neighborNumbers.get(depth).intValue());

                for (Node neighbor : neighbors) {
                    if (visitedIds.add(neighbor.getElementId())) {
                        currentLayerSet.add(neighbor);
                    }
                }
            }

            if (currentLayerSet.isEmpty()) {
                break; // No more neighbors to sample
            }

            layers.add(new ArrayList<>(currentLayerSet));
        }

        return Stream.of(new SampledNodesResult(layers));
    }

    public static class SampledNodesResult {
        public List<List<Node>> layers;
        public List<Node> sampledNodes;

        public SampledNodesResult(List<List<Node>> layers) {
            this.layers = layers;
            this.sampledNodes = layers.stream()
                    .flatMap(List::stream)
                    .collect(Collectors.toList());
        }
    }

}
