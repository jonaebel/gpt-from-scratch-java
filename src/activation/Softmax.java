package activation;

import math.Matrix;

public class Softmax implements ActivationFunction {

    /**
     * Applies softmax row-wise: each row is treated as one probability distribution.
     * softmax(x_i) = exp(x_i) / sum(exp(x_j))
     * Subtracts the row maximum for numerical stability.
     */
    @Override
    public Matrix apply(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++) {
            double max = in[i][0];
            for (int j = 1; j < in[0].length; j++)
                if (in[i][j] > max) max = in[i][j];

            double sum = 0;
            for (int j = 0; j < in[0].length; j++) {
                out[i][j] = Math.exp(in[i][j] - max);
                sum += out[i][j];
            }
            for (int j = 0; j < in[0].length; j++)
                out[i][j] /= sum;
        }
        return new Matrix(out);
    }

    /**
     * Element-wise derivative of softmax: s_i * (1 - s_i).
     * This is the diagonal of the Jacobian and is used in backprop
     * when combined with cross-entropy loss.
     */
    @Override
    public Matrix derivative(Matrix m) {
        double[][] s = apply(m).getValues();
        double[][] out = new double[s.length][s[0].length];
        for (int i = 0; i < s.length; i++)
            for (int j = 0; j < s[0].length; j++)
                out[i][j] = s[i][j] * (1.0 - s[i][j]);
        return new Matrix(out);
    }

    @Override
    public ActivationFunctionType getType() {
        return ActivationFunctionType.SOFTMAX;
    }
}
