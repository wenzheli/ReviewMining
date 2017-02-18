package ml.topicModel.NGramSentence;

import java.util.List;

public class Document{
    List<Sentence> sentences;
    
    public void setSentences(List<Sentence> sentences){
        this.sentences = sentences;
    }
    
    public int getNumOfSentences(){
        return sentences.size();
    }
    
    public Sentence getSentence(int index){
        return sentences.get(index);
    }

    List<Integer> tokens;

    public void setTokens(List<Integer> tokens){
        this.tokens = tokens;
    }

    public int getNumOfTokens(){
        return tokens.size();
    }
    public List<Integer> getTokens(){
        return tokens;
    }

    public int getToken(int index){
        return tokens.get(index);
    }
}
