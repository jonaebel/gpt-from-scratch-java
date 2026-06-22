package network;

import loss.LossFunction;
import math.Matrix;
import optimizer.Optimizer;

public class Network {

    Layer[] layers;
    LossFunction loss;
    Optimizer optimizer;

    public Network(Layer[] layers, LossFunction loss, Optimizer optimizer) {
        this.layers = layers;
        this.loss = loss;
        this.optimizer = optimizer;
    }

    public Matrix forward(Matrix input) {
        Matrix current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    public void backward(Matrix predictions, Matrix targets) {
        // 1. Loss Gradient berechnen
        Matrix grad = loss.gradient(predictions, targets);

        // 2. Rückwärts durch alle Layer
        for (int i = layers.length - 1; i >= 0; i--) {
            grad = layers[i].backwards(grad);
        }
    }

    public void train(Matrix inputs, Matrix targets, int epochs, int batchSize) {
        int numSamples = inputs.getRows();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;
            int numBatches = 0;

            // 1. Daten in Batches aufteilen
            for (int start = 0; start < numSamples; start += batchSize) {
                int end = Math.min(start + batchSize, numSamples);
                Matrix batchInputs = sliceRows(inputs, start, end);
                Matrix batchTargets = sliceRows(targets, start, end);

                // 2. forward → backward → optimizer.step()
                Matrix predictions = forward(batchInputs);
                epochLoss += loss.compute(predictions, batchTargets);
                backward(predictions, batchTargets);
                optimizer.step(layers);

                numBatches++;
            }

            // 3. Loss pro Epoch loggen
            System.out.printf("Epoch %4d / %d  |  Loss: %.6f%n", epoch + 1, epochs, epochLoss / numBatches);
        }
    }

    private Matrix sliceRows(Matrix m, int fromRow, int toRow) {
        int cols = m.getColumns();
        double[][] slice = new double[toRow - fromRow][cols];
        for (int i = fromRow; i < toRow; i++) {
            for (int j = 0; j < cols; j++) {
                slice[i - fromRow][j] = m.get(i, j);
            }
        }
        return new Matrix(slice);
    }
}
