package initialization;

import math.Matrix;

import java.util.Random;

/*
    Xavier (Glorot) Normal initialization: weights ~ N(0, sqrt(2 / (fanIn + fanOut)))
    Designed for layers with sigmoid or tanh activations.
 */
public class XavierNormal implements WeightInitialization {

    private final Random random = new Random();

    @Override
    public Matrix compute(int fanIn, int fanOut) {
        double stddev = Math.sqrt(2.0 / (fanIn + fanOut));
        double[][] values = new double[fanIn][fanOut];
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                values[i][j] = random.nextGaussian() * stddev;
        return new Matrix(values);
    }

    @Override
    public WeightInitializationTyp getTyp() {
        return WeightInitializationTyp.XAVIER_NORMAL;
    }
}
