package network;

import math.Matrix;

import java.util.Random;

// Lookup-Tabelle: Token ID → Vektor
// Unterstützt Batch-Verarbeitung: int[batchSize][seqLen] → Matrix[batchSize][seqLen * embeddingDim]
public class EmbeddingLayer {

    Matrix embeddings;     // [vocabSize][embeddingDim]
    Matrix gradEmbeddings; // [vocabSize][embeddingDim] Gradienten

    private final int vocabSize;
    private final int embeddingDim;

    // gespeicherte Token-IDs aus dem letzten forward() für backward()
    private int[][] lastInputIds; // [batchSize][seqLen]

    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this.vocabSize    = vocabSize;
        this.embeddingDim = embeddingDim;

        gradEmbeddings = new Matrix(vocabSize, embeddingDim);

        Random rng = new Random();
        double[][] values = new double[vocabSize][embeddingDim];
        for (int i = 0; i < vocabSize; i++)
            for (int j = 0; j < embeddingDim; j++)
                values[i][j] = rng.nextGaussian() * 0.01;
        embeddings = new Matrix(values);
    }

    // batchIds: [batchSize][seqLen]
    // return:   [batchSize][seqLen * embeddingDim] alle Token-Vektoren pro Sample hintereinander
    public Matrix forward(int[][] batchIds) {
        lastInputIds = batchIds;
        int batchSize = batchIds.length;
        int seqLen    = batchIds[0].length;
        double[][] embValues = embeddings.getValues();
        double[][] output    = new double[batchSize][seqLen * embeddingDim];

        for (int b = 0; b < batchSize; b++)
            for (int t = 0; t < seqLen; t++)
                System.arraycopy(embValues[batchIds[b][t]], 0,
                                 output[b], t * embeddingDim, embeddingDim);

        return new Matrix(output);
    }

    // gradOutput: [batchSize][seqLen * embeddingDim]
    // Addiert Gradienten auf die Token-Zeilen die im Forward benutzt wurden
    public void backward(Matrix gradOutput) {
        double[][] gradValues    = gradOutput.getValues();
        double[][] gradEmbValues = gradEmbeddings.getValues();
        int seqLen = lastInputIds[0].length;

        for (int b = 0; b < lastInputIds.length; b++)
            for (int t = 0; t < seqLen; t++) {
                int tokenId = lastInputIds[b][t];
                for (int j = 0; j < embeddingDim; j++)
                    // += weil dasselbe Token mehrmals im Batch/Context vorkommen kann und so der trainings effekt wirkt
                    gradEmbValues[tokenId][j] += gradValues[b][t * embeddingDim + j];
            }
    }

    public Matrix getEmbeddings()     { return embeddings; }
    public Matrix getGradEmbeddings() { return gradEmbeddings; }

    public void setEmbeddings(Matrix embeddings) { this.embeddings = embeddings; }
}
