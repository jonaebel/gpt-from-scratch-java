package activation;

import math.Matrix;

public class LeakyReLu implements ActivationFunction {

    private final double alpha;

    public LeakyReLu(double alpha) {
        this.alpha = alpha;
    }

    public LeakyReLu() {
        this(0.01);
    }

    @Override
    public Matrix apply(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[i][j] = in[i][j] > 0 ? in[i][j] : alpha * in[i][j];
        return new Matrix(out);
    }

    @Override
    public Matrix derivative(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[i][j] = in[i][j] > 0 ? 1 : alpha;
        return new Matrix(out);
    }

    @Override
    public ActivationFunctionType getType() {
        return ActivationFunctionType.LEAKY_RELU;
    }
}
