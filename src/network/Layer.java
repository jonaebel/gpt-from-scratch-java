package network;

import activation.ActivationFunction;
import initialization.WeightInitialization;
import math.Matrix;
import math.MatrixMultiplication;

public class Layer {

    // Learning params
    Matrix weights;   // [outputSize][inputSize]
    Matrix weightsT;  // [inputSize][outputSize] — cached transpose, refreshed after each optimizer step
    double[] biases;  // [outputSize]

    // saved during forward()
    Matrix lastInput; // [batchSize][inputSize]
    Matrix lastZ;     // [batchSize][outputSize]  pre-activation

    // Gradients — filled during backwards()
    Matrix gradWeights;  // [outputSize][inputSize]
    double[] gradBiases; // [outputSize]

    ActivationFunction activation;

    public Layer(int inputSize, int outputSize, ActivationFunction activation, WeightInitialization init) {
        this.weights    = init.compute(inputSize, outputSize).transpose();
        this.weightsT   = this.weights.transpose();  // initial cache
        this.biases     = new double[outputSize];
        this.activation = activation;
        this.gradWeights = new Matrix(outputSize, inputSize);
        this.gradBiases  = new double[outputSize];
    }

    /**
     * Refreshes the cached transposed weight matrix.
     * Must be called once after each optimizer step.
     */
    public void recomputeTranspose() {
        weightsT = weights.transpose();
    }

    public Matrix forward(Matrix input) {
        this.lastInput = input;
        // weights is [outSize × inSize] — multiplyBT computes input @ weights^T
        // without allocating a transposed copy (zero-allocation hot path).
        lastZ = MatrixMultiplication.multiplyBT(input, weights).addVector(biases);
        return activation.apply(lastZ);
    }

    public Matrix backwards(Matrix gradOutput) {
        int batchSize = lastInput.getRows();

        Matrix delta = gradOutput.hadamard(activation.derivative(lastZ));

        // gradWeights = delta^T @ lastInput / batchSize
        // multiplyATB avoids allocating delta.transpose()
        gradWeights = MatrixMultiplication.multiplyATB(delta, lastInput).multiply(1.0 / batchSize);

        // bias gradients
        double[][] dv = delta.getValues();
        for (int i = 0; i < gradBiases.length; i++) {
            double sum = 0;
            for (int b = 0; b < batchSize; b++) sum += dv[b][i];
            gradBiases[i] = sum / batchSize;
        }

        // Propagate gradient to the previous layer: delta @ weights
        // weightsT is [inSize × outSize], so multiplyBT computes delta @ weightsT^T = delta @ weights
        return MatrixMultiplication.multiplyBT(delta, weightsT);
    }

    public Matrix getWeights()           { return weights; }
    public void   setWeights(Matrix w)   { this.weights = w; }

    public double[] getBiases()          { return biases; }
    public void     setBiases(double[] b){ this.biases = b; }

    public Matrix   getGradWeights()     { return gradWeights; }
    public double[] getGradBiases()      { return gradBiases; }
}
