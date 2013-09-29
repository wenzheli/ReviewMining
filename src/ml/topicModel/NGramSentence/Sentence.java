package ml.topicModel.NGramSentence;

import java.util.List;

public class Sentence {
    List<Integer> tokens;
    
    public void setTokens(List<Integer> tokens){
        this.tokens = tokens;
    }
    
    public List<Integer> getTokens(){
        return tokens;
    }
    
    public int getToken(int idx){
        return tokens.get(idx);
    }
}
