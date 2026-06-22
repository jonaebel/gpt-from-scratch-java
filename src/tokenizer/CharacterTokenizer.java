package tokenizer;

import java.util.HashMap;
import java.util.Map;

public class CharacterTokenizer implements Tokenizer{

    private final Map<Character, Integer> charToId; // 'a' → 3
    private final Map<Integer, Character> idToChar; // 3 → 'a'
    private final int vocabSize;

    // Baut das Vokabular aus dem übergebenen Text auf
    // Jedes einzigartige Zeichen bekommt eine ID
    public CharacterTokenizer(String text) {
        charToId = new HashMap<>();
        idToChar = new HashMap<>();

        int id = 0;
        for (char c : text.toCharArray()) {
            if (!charToId.containsKey(c)) {
                charToId.put(c, id);
                idToChar.put(id, c);
                id++;
            }
        }
        this.vocabSize = id;
    }

    @Override
    public int[] encode(String text) {
        int[] tokens = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            tokens[i] = charToId.get(text.charAt(i));
        }
        return tokens;
    }

    @Override
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            sb.append(idToChar.get(token));
        }
        return sb.toString();
    }

    @Override
    public TokenizerType getType() {
        return TokenizerType.CHARACKTERTOKENIZER;
    }

    @Override
    public int vocabSize() {
        return vocabSize;
    }
}
