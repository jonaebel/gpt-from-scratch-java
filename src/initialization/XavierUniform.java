package initialization;

import math.Matrix;

import java.util.Random;

/*
    Xavier (Glorot) Uniform initialization: weights ~ U(-sqrt(6 / (fanIn + fanOut)), sqrt(6 / (fanIn + fanOut)))
    Designed for layers with sigmoid or tanh activations.
 */
public class XavierUniform implements WeightInitialization {

    private final Random random = new Random();

    @Override
    public Matrix compute(int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        double[][] values = new double[fanIn][fanOut];
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                values[i][j] = random.nextDouble() * 2 * limit - limit;
        return new Matrix(values);
    }

    @Override
    public WeightInitializationTyp getTyp() {
        return WeightInitializationTyp.XAVIER_UNIFORM;
    }
}
