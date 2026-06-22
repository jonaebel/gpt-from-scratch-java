package optimizer;

import math.Matrix;
import network.Layer;

/*
    Stochastic Gradient Descent (SGD) with optional momentum
        Update without momentum: w = w - lr * gradW
        Update with momentum:    v = momentum * v - lr * gradW
                                 w = w + v
        A higher momentum (e.g. 0.9) smooths updates and can speed up convergence.
        Set momentum to 0 for vanilla SGD.
 */
public class SDG implements Optimizer {

    private final double learningRate;
    private final double momentum;

    // velocity matrices per layer (for momentum)
    private Matrix[] velocityWeights;
    private double[][] velocityBiases;

    public SDG(double learningRate) {
        this(learningRate, 0.0);
    }

    public SDG(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public void step(Layer[] layers) {
        if (velocityWeights == null)
            initVelocities(layers);

        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            double[][] w = layer.getWeights().getValues();
            double[][] gw = layer.getGradWeights().getValues();
            double[] b = layer.getBiases();
            double[] gb = layer.getGradBiases();

            // update weights
            double[][] newW = new double[w.length][w[0].length];
            double[][] vw = velocityWeights[l].getValues();
            for (int i = 0; i < w.length; i++)
                for (int j = 0; j < w[0].length; j++) {
                    vw[i][j] = momentum * vw[i][j] - learningRate * gw[i][j];
                    newW[i][j] = w[i][j] + vw[i][j];
                }
            layer.setWeights(new Matrix(newW));
            velocityWeights[l] = new Matrix(vw);

            // update biases
            double[] newB = new double[b.length];
            for (int i = 0; i < b.length; i++) {
                velocityBiases[l][i] = momentum * velocityBiases[l][i] - learningRate * gb[i];
                newB[i] = b[i] + velocityBiases[l][i];
            }
            layer.setBiases(newB);
        }
    }

    private void initVelocities(Layer[] layers) {
        velocityWeights = new Matrix[layers.length];
        velocityBiases = new double[layers.length][];
        for (int l = 0; l < layers.length; l++) {
            double[][] w = layers[l].getWeights().getValues();
            velocityWeights[l] = new Matrix(w.length, w[0].length);
            velocityBiases[l] = new double[layers[l].getBiases().length];
        }
    }

    @Override
    public OptimizerType getTyp() {
        return OptimizerType.SGD;
    }
}
