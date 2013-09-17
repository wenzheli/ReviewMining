package ml.topicModel.ngram;

import java.util.List;

public class NGramDocument {
    List<List<Integer>> terms;
    List<Integer> topics;
    
    public void setTerms(List<List<Integer>> terms){
        this.terms = terms;
    }
    
    public void setTopics(List<Integer> topics){
        this.topics = topics;
    }
    
    public List<List<Integer>> getTerms(){
        return terms;
    }
    
    public List<Integer> getTopics(){
        return topics;
    }
}
