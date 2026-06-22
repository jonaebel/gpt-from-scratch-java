package utils;

import math.Matrix;
import network.EmbeddingLayer;
import network.Layer;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/*
    Speichert und lädt den kompletten Modellzustand binär.

    Dateiformat:
        [int]    vocabSize
        [char × vocabSize]                      Vokabular
        [int]    embeddingDim
        [double × vocabSize × embeddingDim]     Embedding-Gewichte
        [int]    numLayers
        pro Layer:
            [int]    outputSize
            [int]    inputSize
            [double × outputSize × inputSize]   Gewichte
            [double × outputSize]               Biases
 */
public class ModelMemory {

    public static class ModelState {
        public final List<Character> vocab;
        public final EmbeddingLayer  embedding;
        public final Layer[]         layers;

        public ModelState(List<Character> vocab, EmbeddingLayer embedding, Layer[] layers) {
            this.vocab     = vocab;
            this.embedding = embedding;
            this.layers    = layers;
        }
    }

    public static void save(String path,
                            List<Character> vocab,
                            EmbeddingLayer embedding,
                            Layer[] layers) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(Path.of(path))))) {

            // Vokabular
            out.writeInt(vocab.size());
            for (char c : vocab) out.writeChar(c);

            // Embedding
            double[][] emb = embedding.getEmbeddings().getValues();
            out.writeInt(emb[0].length);
            for (double[] row : emb)
                for (double v : row) out.writeDouble(v);

            // Dense-Layer
            out.writeInt(layers.length);
            for (Layer layer : layers) {
                double[][] w = layer.getWeights().getValues();
                double[]   b = layer.getBiases();
                out.writeInt(w.length);
                out.writeInt(w[0].length);
                for (double[] row : w)
                    for (double v : row) out.writeDouble(v);
                for (double v : b) out.writeDouble(v);
            }
        }
        System.out.printf("[ModelMemory] Gespeichert → %s (%.1f KB)%n",
            path, Files.size(Path.of(path)) / 1024.0);
    }

    // Gibt null zurück wenn keine Datei existiert
    public static ModelState load(String path) throws IOException {
        if (!Files.exists(Path.of(path))) return null;

        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(Path.of(path))))) {

            // Vokabular
            int vocabSize = in.readInt();
            List<Character> vocab = new ArrayList<>(vocabSize);
            for (int i = 0; i < vocabSize; i++) vocab.add(in.readChar());

            // Embedding
            int embeddingDim = in.readInt();
            double[][] embValues = new double[vocabSize][embeddingDim];
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < embeddingDim; j++)
                    embValues[i][j] = in.readDouble();
            EmbeddingLayer embedding = new EmbeddingLayer(vocabSize, embeddingDim);
            embedding.setEmbeddings(new Matrix(embValues));

            // Dense-Layer
            int numLayers = in.readInt();
            Layer[] layers = new Layer[numLayers];
            for (int l = 0; l < numLayers; l++) {
                int outputSize = in.readInt();
                int inputSize  = in.readInt();
                double[][] w = new double[outputSize][inputSize];
                for (int i = 0; i < outputSize; i++)
                    for (int j = 0; j < inputSize; j++)
                        w[i][j] = in.readDouble();
                double[] b = new double[outputSize];
                for (int i = 0; i < outputSize; i++) b[i] = in.readDouble();
                layers[l] = new Layer(inputSize, outputSize,
                                      new activation.Linear(),
                                      new initialization.XavierUniform());
                layers[l].setWeights(new Matrix(w));
                layers[l].setBiases(b);
            }

            System.out.printf("[ModelMemory] Geladen ← %s  (%d Zeichen Vokabular, %d Layer)%n",
                path, vocabSize, numLayers);
            return new ModelState(vocab, embedding, layers);
        }
    }
}
