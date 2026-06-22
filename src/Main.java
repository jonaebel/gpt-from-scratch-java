import activation.Linear;
import initialization.XavierUniform;
import loss.SoftmaxCCE;
import math.Matrix;
import network.EmbeddingLayer;
import network.Layer;
import network.TransformerBlock;
import optimizer.Adam;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/*
 * Transformer-basiertes Zeichen-Sprachmodell
 *
 * Architektur pro Config:
 *
 *   EmbeddingLayer  (vocabSize × embDim)
 *   + Positionale Embeddings  (seqLen × embDim)   — gelernt, addiert
 *   N × TransformerBlock      (LayerNorm+MHA+FFN, Pre-Norm, Causal)
 *   Layer (Linear)            (embDim → vocabSize)
 *
 * Trainingsschleife pro Batch:
 *   1. Embedding + PosEmb → [B × seqLen × embDim]  (logisch 3D, verarbeitet per Sample)
 *   2. Transformer-Blöcke  pro Sample [seqLen × embDim] → [seqLen × embDim]
 *   3. Letztes Token       → [B × embDim]
 *   4. Output-Layer        → [B × vocabSize]  →  SoftmaxCCE-Loss
 *   5. Rückwärts:          Gradienten akkumulieren über den Batch
 *   6. Adam-Schritte:      Transformer-Blöcke, Embedding, PosEmb, Output-Layer
 */
public class Main {

    // ── Konfiguration ─────────────────────────────────────────────────────────

    static final String CONFIG = "1M";   // "1M", "5M" oder "25M"

    static final int    SEQ_LEN     = 64;
    static final int    BATCH_SIZE  = 32;
    static final int    TRAIN_STEPS = 5_000;
    static final double LR          = 3e-4;
    static final double TEMPERATURE = 0.8;
    static final int    LOG_EVERY   = 200;

    // Architektur-Parameter je CONFIG
    static final int EMB_DIM;
    static final int NUM_HEADS;
    static final int NUM_BLOCKS;

    static {
        switch (CONFIG) {
            case "1M"  -> { EMB_DIM = 64;  NUM_HEADS = 4; NUM_BLOCKS = 2; }
            case "5M"  -> { EMB_DIM = 192; NUM_HEADS = 8; NUM_BLOCKS = 6; }
            default    -> { EMB_DIM = 384; NUM_HEADS = 8; NUM_BLOCKS = 12; } // "25M"
        }
    }

    // ── Adam-Zustand für Embedding und PosEmb ────────────────────────────────

    static double[][] embM, embV;
    static int        embT = 0;

    static double[][] posM, posV;

    // ── Main ──────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws IOException {

        // 1. Text laden
        Path trainPath = Files.exists(Path.of("train.txt"))
                       ? Path.of("train.txt") : Path.of("wikipedia.txt");
        System.out.println("Trainingsdatei : " + trainPath);
        String text = Files.readString(trainPath);
        text = text.substring(0, Math.min(100_000_000, text.length()));

        // 2. Vokabular aufbauen
        List<Character>      vocab    = new ArrayList<>(new TreeSet<>(toCharList(text)));
        Map<Character, Integer> c2i   = new HashMap<>();
        Map<Integer, Character> i2c   = new HashMap<>();
        for (int i = 0; i < vocab.size(); i++) { c2i.put(vocab.get(i), i); i2c.put(i, vocab.get(i)); }
        int vocabSize = vocab.size();
        int spaceId   = c2i.getOrDefault(' ', 0);

        int[] textIds = new int[text.length()];
        for (int i = 0; i < text.length(); i++) textIds[i] = c2i.get(text.charAt(i));

        // 3. Modell aufbauen
        EmbeddingLayer    embedding = new EmbeddingLayer(vocabSize, EMB_DIM);
        double[][]        posEmb    = randomArray(SEQ_LEN, EMB_DIM, 0.02); // [seqLen × embDim]
        TransformerBlock[] blocks   = new TransformerBlock[NUM_BLOCKS];
        for (int i = 0; i < NUM_BLOCKS; i++) blocks[i] = new TransformerBlock(EMB_DIM, NUM_HEADS);
        Layer   outputLayer  = new Layer(EMB_DIM, vocabSize, new Linear(), new XavierUniform());
        Adam    outputAdam   = new Adam(LR);

        long params = countParams(vocabSize, blocks, outputLayer);
        System.out.printf("Konfiguration  : %s  (~%,d Parameter)%n", CONFIG, params);
        System.out.printf("Vokabulargröße : %d Zeichen%n", vocabSize);
        System.out.printf("EMB=%d  HEADS=%d  BLOCKS=%d  SEQ=%d%n%n",
                          EMB_DIM, NUM_HEADS, NUM_BLOCKS, SEQ_LEN);

        // 4. Benutzerabfrage
        Scanner scanner    = new Scanner(System.in);
        System.out.print("Trainieren? (j/n): ");
        boolean doTraining = scanner.nextLine().trim().equalsIgnoreCase("j");

        SoftmaxCCE loss = new SoftmaxCCE();
        Random     rng  = new Random(42);

        // 5. Training
        if (doTraining) {
            System.out.println("--- Training ---");
            double runningLoss = 0;
            int    maxStart    = textIds.length - SEQ_LEN - 1;

            for (int step = 1; step <= TRAIN_STEPS; step++) {

                // ── Batch aufbauen ────────────────────────────────────────────
                // Jede Position t in jedem Sample ist ein eigenes Trainingsziel:
                //   Input[b][t] → Ziel: textIds[pos + t + 1]
                // Das gibt BATCH_SIZE × SEQ_LEN Vorhersagen pro Schritt statt nur BATCH_SIZE.
                int[][]    batchIds   = new int[BATCH_SIZE][SEQ_LEN];
                int[]      startPos   = new int[BATCH_SIZE];
                // Targets: [B*SEQ_LEN × vocabSize] — alle Positionen aller Samples
                double[][] targetData = new double[BATCH_SIZE * SEQ_LEN][vocabSize];
                for (int b = 0; b < BATCH_SIZE; b++) {
                    startPos[b]  = rng.nextInt(maxStart);
                    batchIds[b]  = Arrays.copyOfRange(textIds, startPos[b], startPos[b] + SEQ_LEN);
                    for (int t = 0; t < SEQ_LEN; t++)
                        targetData[b * SEQ_LEN + t][textIds[startPos[b] + t + 1]] = 1.0;
                }
                Matrix targets = new Matrix(targetData);

                // ── Forward ───────────────────────────────────────────────────
                // Embedding: [B × seqLen*embDim] (flat)
                Matrix embFlat = embedding.forward(batchIds);

                // Pro Sample: reshape + PosEmb + Transformer-Blöcke → ALLE Token-Outputs
                // Gesammelt in [B*SEQ_LEN × embDim]
                double[][] allTokenVecs = new double[BATCH_SIZE * SEQ_LEN][EMB_DIM];

                for (int b = 0; b < BATCH_SIZE; b++) {
                    Matrix x = addPosEmb(embFlat.getValues()[b], posEmb);
                    for (TransformerBlock block : blocks) x = block.forward(x);
                    double[][] xv = x.getValues(); // [SEQ_LEN × EMB_DIM]
                    for (int t = 0; t < SEQ_LEN; t++)
                        allTokenVecs[b * SEQ_LEN + t] = xv[t].clone();
                }

                // Output-Layer: [B*SEQ_LEN × embDim] → [B*SEQ_LEN × vocabSize]
                Matrix allTok = new Matrix(allTokenVecs);
                Matrix pred   = outputLayer.forward(allTok);

                runningLoss += loss.compute(pred, targets);

                // ── Backward ─────────────────────────────────────────────────
                Matrix grad   = loss.gradient(pred, targets);    // [B*SEQ_LEN × vocabSize]
                Matrix gradAT = outputLayer.backwards(grad);     // [B*SEQ_LEN × embDim]

                // Transformer-Gradienten auf 0 setzen (Akkumulation beginnt)
                for (TransformerBlock block : blocks) block.zeroGradients();

                double[][] posGrad  = new double[SEQ_LEN][EMB_DIM];
                double[][] embGrads = new double[BATCH_SIZE][SEQ_LEN * EMB_DIM];

                double[][] gradATv = gradAT.getValues(); // [B*SEQ_LEN × embDim]
                for (int b = 0; b < BATCH_SIZE; b++) {
                    // Gradient für alle SEQ_LEN Positionen dieses Samples rekonstruieren
                    double[][] sampleGradData = new double[SEQ_LEN][EMB_DIM];
                    for (int t = 0; t < SEQ_LEN; t++)
                        sampleGradData[t] = gradATv[b * SEQ_LEN + t];
                    Matrix sampleGrad = new Matrix(sampleGradData);

                    // Rückwärts durch alle Transformer-Blöcke
                    for (int i = blocks.length - 1; i >= 0; i--)
                        sampleGrad = blocks[i].backward(sampleGrad);

                    // sampleGrad ist jetzt dX = dEmbedding + dPosEmb für alle Positionen
                    double[][] sgv = sampleGrad.getValues();
                    for (int t = 0; t < SEQ_LEN; t++) {
                        for (int j = 0; j < EMB_DIM; j++) posGrad[t][j] += sgv[t][j];
                        System.arraycopy(sgv[t], 0, embGrads[b], t * EMB_DIM, EMB_DIM);
                    }
                }

                // Embedding-Rückwärtspass (akkumuliert intern)
                embedding.backward(new Matrix(embGrads));

                // ── Optimizer-Schritte ────────────────────────────────────────
                // Output-Layer (eigener Adam, verarbeitet Batch-Matrix intern)
                outputAdam.step(new Layer[]{outputLayer});
                outputLayer.recomputeTranspose();

                // Transformer-Blöcke (Adam pro Block, normiert durch BATCH_SIZE)
                embT++;
                for (TransformerBlock block : blocks)
                    block.adamStep(LR, 0.9, 0.999, 1e-8, embT, BATCH_SIZE);

                // Embedding (Adam, wie bisher)
                updateEmbedding(embedding, LR);

                // Positionale Embeddings (Adam, teilt sich embT-Zähler)
                updatePosEmb(posEmb, posGrad, LR, BATCH_SIZE);

                // ── Logging ───────────────────────────────────────────────────
                if (step % LOG_EVERY == 0) {
                    System.out.printf("  Schritt %6d / %,d  |  Ø Loss: %.4f%n",
                                      step, TRAIN_STEPS, runningLoss / LOG_EVERY);
                    runningLoss = 0;
                }
            }
            System.out.println("Training abgeschlossen.");
        }

        // 6. Textgenerierung
        System.out.print("\nEingabewort: ");
        String word = scanner.nextLine().trim();

        System.out.print("Anzahl Zeichen generieren: ");
        int genCount = 500;
        try { genCount = Integer.parseInt(scanner.nextLine().trim()); }
        catch (NumberFormatException ignored) {}

        // Kontext aufbauen (links mit Leerzeichen auffüllen)
        String padded = String.format("%" + SEQ_LEN + "s", word);
        padded = padded.substring(padded.length() - SEQ_LEN);
        int[] ctx = new int[SEQ_LEN];
        for (int i = 0; i < SEQ_LEN; i++) ctx[i] = c2i.getOrDefault(padded.charAt(i), spaceId);

        System.out.println("\n--- Generierter Text ---");
        System.out.print(word);

        for (int i = 0; i < genCount; i++) {
            // Forward: ein Sample [1 × seqLen*embDim]
            Matrix ef = embedding.forward(new int[][]{ctx});
            Matrix x  = addPosEmb(ef.getValues()[0], posEmb);
            for (TransformerBlock block : blocks) x = block.forward(x);

            // Letztes Token → Softmax → Sampling
            double[] logits  = x.getValues()[SEQ_LEN - 1];
            Matrix   logMat  = new Matrix(new double[][]{logits});
            Matrix   probs   = SoftmaxCCE.softmax(logMat);
            int      nextId  = sample(probs, TEMPERATURE, rng);

            System.out.print(i2c.get(nextId));
            System.out.flush();

            // Kontext verschieben
            System.arraycopy(ctx, 1, ctx, 0, SEQ_LEN - 1);
            ctx[SEQ_LEN - 1] = nextId;
        }
        System.out.println();
    }

    // ── Hilfsmethoden ─────────────────────────────────────────────────────────

    /**
     * Nimmt den flachen Embedding-Vektor eines Samples [seqLen*embDim]
     * und addiert die lernbaren positionalen Embeddings zeilenweise.
     * Gibt eine [seqLen × embDim] Matrix zurück.
     */
    private static Matrix addPosEmb(double[] flat, double[][] posEmb) {
        int seqLen = posEmb.length, embDim = posEmb[0].length;
        double[][] out = new double[seqLen][embDim];
        for (int t = 0; t < seqLen; t++)
            for (int j = 0; j < embDim; j++)
                out[t][j] = flat[t * embDim + j] + posEmb[t][j];
        return new Matrix(out);
    }

    /** Adam-Update für das Embedding (identisch zur bisherigen Implementierung). */
    private static void updateEmbedding(EmbeddingLayer emb, double lr) {
        double[][] w = emb.getEmbeddings().getValues();
        double[][] g = emb.getGradEmbeddings().getValues();
        if (embM == null) { embM = new double[w.length][w[0].length]; embV = new double[w.length][w[0].length]; }

        double bc1 = 1 - Math.pow(0.9,   embT);
        double bc2 = 1 - Math.pow(0.999, embT);
        for (int i = 0; i < w.length; i++)
            for (int j = 0; j < w[0].length; j++) {
                double grad = g[i][j];
                embM[i][j] = 0.9 * embM[i][j] + 0.1 * grad;
                embV[i][j] = 0.999 * embV[i][j] + 0.001 * grad * grad;
                w[i][j]   -= lr * (embM[i][j] / bc1) / (Math.sqrt(embV[i][j] / bc2) + 1e-8);
                g[i][j]    = 0;  // Gradient zurücksetzen
            }
    }

    /** Adam-Update für die positionalen Embeddings. */
    private static void updatePosEmb(double[][] posEmb, double[][] posGrad, double lr, int batchSize) {
        int seqLen = posEmb.length, embDim = posEmb[0].length;
        if (posM == null) { posM = new double[seqLen][embDim]; posV = new double[seqLen][embDim]; }

        double bc1 = 1 - Math.pow(0.9,   embT);
        double bc2 = 1 - Math.pow(0.999, embT);
        for (int t = 0; t < seqLen; t++)
            for (int j = 0; j < embDim; j++) {
                double g = posGrad[t][j] / batchSize;
                posM[t][j] = 0.9 * posM[t][j] + 0.1 * g;
                posV[t][j] = 0.999 * posV[t][j] + 0.001 * g * g;
                posEmb[t][j] -= lr * (posM[t][j] / bc1) / (Math.sqrt(posV[t][j] / bc2) + 1e-8);
                posGrad[t][j] = 0;
            }
    }

    /** Temperature-Sampling. */
    private static int sample(Matrix probs, double temperature, Random rng) {
        double[] p   = probs.getValues()[0];
        double[] sc  = new double[p.length];
        double   sum = 0;
        for (int i = 0; i < p.length; i++) {
            sc[i] = Math.exp(Math.log(Math.max(p[i], 1e-12)) / temperature);
            sum  += sc[i];
        }
        double draw = rng.nextDouble() * sum;
        for (int i = 0; i < sc.length; i++) { draw -= sc[i]; if (draw <= 0) return i; }
        return sc.length - 1;
    }

    /** Parameter zählen (Näherung). */
    private static long countParams(int vocabSize, TransformerBlock[] blocks, Layer out) {
        long total = (long) vocabSize * EMB_DIM;                       // Embedding
        total     += (long) SEQ_LEN * EMB_DIM;                         // PosEmb
        for (TransformerBlock b : blocks) {
            int d = EMB_DIM, f = 4 * d, h = b.getAttention().getHeads().length;
            int dk = d / h;
            total += (long) h * 3 * dk * d;                            // W_Q, W_K, W_V
            total += (long) d * d;                                      // W_O
            total += (long) f * d + d * f;                              // W1, W2
            total += f + d + 4 * d;                                     // b1, b2, γ, β ×2
        }
        double[][] w = out.getWeights().getValues();
        total += (long) w.length * w[0].length + w.length;             // Output-Layer
        return total;
    }

    /** Zufällige double[rows][cols]-Initialisierung. */
    private static double[][] randomArray(int rows, int cols, double std) {
        Random rng = new Random();
        double[][] v = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                v[i][j] = rng.nextGaussian() * std;
        return v;
    }

    private static List<Character> toCharList(String s) {
        List<Character> list = new ArrayList<>(s.length());
        for (char c : s.toCharArray()) list.add(c);
        return list;
    }
}
