package tokenizer;

public interface Tokenizer {

    int[] encode(String text);
    String decode(int[] token);
    int vocabSize();

    TokenizerType getType();

}
