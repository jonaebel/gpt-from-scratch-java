package loss;

import math.Matrix;

/*
    Kombinierter Softmax + Categorical Cross-Entropy Loss.

    Warum kombiniert?
        Softmax.derivative() liefert nur die Diagonale s*(1-s) des echten Jacobians
        (diag(s) - s*s^T). Wenn CCE und Softmax getrennt durch die Chain-Rule laufen,
        werden alle Gradienten für falsche Klassen auf 0 gesetzt — das Modell lernt nichts.

    Korrekte kombinierte Ableitung (d(CCE∘Softmax)/dz):
        grad_i = (softmax(z)_i - target_i) / N

    Verwendung: Output-Layer mit Linear-Aktivierung + dieser Loss.
        forward : Layer(Linear) → SoftmaxCCE.compute()
        backward: SoftmaxCCE.gradient() → Layer.backwards()  (Linear-Ableitung = 1, kein Effekt)
 */
public class SoftmaxCCE implements LossFunction {

    private static final double EPS = 1e-12;

    @Override
    public double compute(Matrix logits, Matrix targets) {
        Matrix probs   = softmax(logits);
        double[][] p   = probs.getValues();
        double[][] t   = targets.getValues();
        int N          = p.length;
        double sum     = 0;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++)
                sum += t[i][j] * Math.log(Math.max(EPS, p[i][j]));
        return -sum / N;
    }

    // Gradient: (softmax(logits) - targets) / N
    @Override
    public Matrix gradient(Matrix logits, Matrix targets) {
        Matrix   probs = softmax(logits);
        double[][] p   = probs.getValues();
        double[][] t   = targets.getValues();
        int N          = p.length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++)
                grad[i][j] = (p[i][j] - t[i][j]) / N;
        return new Matrix(grad);
    }

    // Zeilenweise numerisch stabile Softmax (wird auch in compute/gradient genutzt)
    public static Matrix softmax(Matrix m) {
        double[][] in  = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++) {
            double max = in[i][0];
            for (double v : in[i]) if (v > max) max = v;
            double sum = 0;
            for (int j = 0; j < in[i].length; j++) {
                out[i][j] = Math.exp(in[i][j] - max);
                sum += out[i][j];
            }
            for (int j = 0; j < in[i].length; j++)
                out[i][j] /= sum;
        }
        return new Matrix(out);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.SOFTMAX_CCE;
    }
}
