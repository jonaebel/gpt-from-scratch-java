package optimizer;

import math.Matrix;
import network.Layer;

import java.util.stream.IntStream;

/*
    Adam (Adaptive Moment Estimation)
        m  = beta1 * m + (1 - beta1) * grad           (first moment)
        v  = beta2 * v + (1 - beta2) * grad^2         (second moment)
        m^ = m / (1 - beta1^t)                        (bias-corrected)
        v^ = v / (1 - beta2^t)                        (bias-corrected)
        w  = w - lr * m^ / (sqrt(v^) + epsilon)
        Typical defaults: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
 */
public class Adam implements Optimizer {

    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private int t = 0;

    // first moment (mean of gradients)
    private double[][][] mWeights;
    private double[][] mBiases;

    // second moment (uncentered variance of gradients)
    private double[][][] vWeights;
    private double[][] vBiases;

    public Adam() {
        this(0.001, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void step(Layer[] layers) {
        if (mWeights == null)
            initMoments(layers);

        t++;
        final double bc1 = 1.0 - Math.pow(beta1, t);
        final double bc2 = 1.0 - Math.pow(beta2, t);

        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            final double[][] w  = layer.getWeights().getValues();
            final double[][] gw = layer.getGradWeights().getValues();
            final double[][] mw = mWeights[l];
            final double[][] vw = vWeights[l];

            // Update weights in-place — parallel over rows (avoids per-step allocation)
            IntStream.range(0, w.length).parallel().forEach(i -> {
                double[] wi  = w[i];
                double[] gwi = gw[i];
                double[] mwi = mw[i];
                double[] vwi = vw[i];
                for (int j = 0; j < wi.length; j++) {
                    mwi[j] = beta1 * mwi[j] + (1 - beta1) * gwi[j];
                    vwi[j] = beta2 * vwi[j] + (1 - beta2) * gwi[j] * gwi[j];
                    wi[j] -= learningRate * (mwi[j] / bc1) / (Math.sqrt(vwi[j] / bc2) + epsilon);
                }
            });

            // Update biases (small array, sequential is fine)
            double[] b  = layer.getBiases();
            double[] gb = layer.getGradBiases();
            double[] mb = mBiases[l];
            double[] vb = vBiases[l];
            for (int i = 0; i < b.length; i++) {
                mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i];
                vb[i] = beta2 * vb[i] + (1 - beta2) * gb[i] * gb[i];
                b[i] -= learningRate * (mb[i] / bc1) / (Math.sqrt(vb[i] / bc2) + epsilon);
            }
        }
    }

    private void initMoments(Layer[] layers) {
        mWeights = new double[layers.length][][];
        vWeights = new double[layers.length][][];
        mBiases  = new double[layers.length][];
        vBiases  = new double[layers.length][];
        for (int l = 0; l < layers.length; l++) {
            double[][] w = layers[l].getWeights().getValues();
            int rows = w.length, cols = w[0].length;
            mWeights[l] = new double[rows][cols];
            vWeights[l] = new double[rows][cols];
            mBiases[l]  = new double[layers[l].getBiases().length];
            vBiases[l]  = new double[layers[l].getBiases().length];
        }
    }

    @Override
    public OptimizerType getTyp() {
        return OptimizerType.ADAM;
    }
}
