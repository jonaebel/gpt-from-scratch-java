package activation;

import math.Matrix;

public class ReLu implements ActivationFunction {

    @Override
    public Matrix apply(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[i][j] = in[i][j] > 0 ? in[i][j] : 0;
        return new Matrix(out);
    }

    @Override
    public Matrix derivative(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[i][j] = in[i][j] > 0 ? 1 : 0;
        return new Matrix(out);
    }

    @Override
    public ActivationFunctionType getType() {
        return ActivationFunctionType.RELU;
    }
}
