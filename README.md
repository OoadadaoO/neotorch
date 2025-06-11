# NeoTorch

## Getting Started

### Prerequisites

- Ubuntu 24.04+

### Install OpenJDK

```bash
sudo apt install openjdk-21-headless
```

### Install Neo4j

Follow the [Neo4j installation guide](https://neo4j.com/docs/operations-manual/current/installation/linux/debian/#debian-installation).

Also refer to the [Neo4j systemd service guide](https://neo4j.com/docs/operations-manual/current/installation/linux/systemd/) to set up Neo4j as a system service and set an initial password.

### Install NeoTorch

Clone the repository:

```bash
git clone https://github.com/OoadadaoO/neotorch.git
cd neotorch
```

To prepare the NeoTorch environment,

```bash
sudo bash prepare.sh
```

Then build from source and deploy the plugin:

```bash
make
```

### (Optional) Install Python Dependencies

If you want to use data loaders and experiments, you need to install the Python dependencies at local:

```bash
bash setup_dev.sh
```

## Experiment

```
bash experiment.sh
```

## User-defined Procedures - GraphSAGE

### Train Mode

```js
CALL neotorch.graphsage.train(
    modelName: String,
    nodes: List<Node>,  // use collect()
    configuration: Map<String, String>,
) YEILD
  modelInfo: Map,
  configuration: Map,
  trainMillis: Integer
```

#### `configuration`

| Key                       | Type            | Default   | Description                                                     |
| ------------------------- | --------------- | --------- | --------------------------------------------------------------- |
| **(Model Parameters)**    |                 |           |                                                                 |
| featureProperties         | List\<String\>  | required  | List of feature properties to use for input.                    |
| featureDimension          | Integer         | required  | Dimension of the input features.                                |
| nodeLabels                | List\<String\>  | `['*']`   | List of node labels to apply for model.                         |
| relationshipTypes         | List\<String\>  | `['*']`   | List of relationship types to apply for model.                  |
| supervised                | Boolean         | `false`   | Whether to use supervised learning.                             |
| embeddingDimension        | Integer         | `64`      | _(unsupervised)_ Dimension of the output embedding.             |
| classProperties           | String          | `'y'`     | _(supervised)_ Class property to predict.                       |
| classDimension            | Integer         | `2`       | _(supervised)_ Dimension of the output class.                   |
| hiddenDimension           | Integer         | `128`     | Dimension of the hidden layer.                                  |
| sampleSizes               | List\<Integer\> | `[10, 5]` | Neighbor sample size for each layer.                            |
| aggregator                | String          | `'mean'`  | \*Aggregator function for GNN layers. `['mean', 'max', 'lstm']` |
| activationFunction        | String          | `'relu'`  | \*Activation function for GNN layers. `['relu', 'sigmoid']`     |
| preLinearLayers           | Integer         | `0`       | _(New)_ Number of linear layers before GNN layers.              |
| postLinearLayers          | Integer         | `0`       | _(New)_ Number of linear layers after GNN layers.               |
| dropoutRate               | Float           | `0.0`     | _(New)_ Dropout rate for GNN layers.                            |
| layerNormalization        | Boolean         | `false`   | _(New)_ Whether to apply layer normalization.                   |
| residualConnection        | Boolean         | `false`   | _(New)_ Whether to use residual connections in GNN layers.      |
| **(Training Parameters)** |                 |           |                                                                 |
| maxGpus                   | Integer         | `1`       | Maximum number of GPUs to use for training.                     |
| randomSeed                | Integer         | `null`    | Random seed for reproducibility.                                |
| batchSize                 | Integer         | `100`     | Batch size for training.                                        |
| epochs                    | Integer         | `10`      | Number of epochs for training.                                  |
| optimizer                 | String          | `'adam'`  | Optimizer for training. `['adam', 'sgd']`                       |
| learningRate              | Float           | `0.001`   | Learning rate for training.                                     |
| negativeSampleWeight      | Float           | `1.0`     | Weight for negative samples in loss function.                   |

### Inference Mode

```js
CALL neotorch.graphsage.infer(
    modelName: String,
    nodes: List<Node>,  // use collect()
    configuration: Map<String, String>,
) YEILD
  modelInfo: Map,
  configuration: Map,
  inferenceMillis: Integer
```

#### `configuration`

| Key                        | Type    | Default | Description                                    |
| -------------------------- | ------- | ------- | ---------------------------------------------- |
| **(Inference Parameters)** |         |         |                                                |
| maxGpus                    | Integer | `1`     | Maximum number of GPUs to use for inferencing. |
| randomSeed                 | Integer | `null`  | Random seed for reproducibility.               |
| batchSize                  | Integer | `100`   | Batch size for inferencing.                    |
