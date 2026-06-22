package data;

public class TextDataLoader {
    private final int[] tokens;
    private final int contextSize;

    // text bereits tokenisiert übergeben
    public TextDataLoader(int[] tokens, int contextSize) {
        this.tokens      = tokens;
        this.contextSize = contextSize;
    }

    // Anzahl möglicher Trainingspaare
    public int size() {
        return tokens.length - contextSize;
    }

    // gibt Input-IDs für Beispiel i zurück
    public int[] getInput(int i) {
        int[] input = new int[contextSize];
        for (int j = 0; j < contextSize; j++) {
            input[j] = tokens[i + j];
        }
        return input;
    }

    // gibt Target-ID für Beispiel i zurück — das nächste Token
    public int getTarget(int i) {
        return tokens[i + contextSize];
    }
}
