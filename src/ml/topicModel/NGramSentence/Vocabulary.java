package ml.topicModel.NGramSentence;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ml.topicModel.jointAspectSentiment.SentimentWords;

public class Vocabulary {
    
    Map<String, Integer> tokenToIndexMap = new HashMap<String, Integer>();
    Map<Integer, String> indexTotokenMap = new HashMap<Integer, String>();
        
    Set<Integer> positiveWords = new HashSet<Integer>();
    Set<Integer> negativeWords = new HashSet<Integer>();
    
    public Set<Integer> getPositiveWordS(){
        return positiveWords;
    }
    
    public Set<Integer> getNegativeWords(){
        return negativeWords;
    }
    
    public void settokenToIndex(Map<String, Integer> tokenToIndexMap){
        this.tokenToIndexMap = tokenToIndexMap;
    }
    
    public void setIndexTotokenMap(Map<Integer, String> indexTotokenMap){
        this.indexTotokenMap = indexTotokenMap;
    }
    
    public int getVocabularySize(){
        return tokenToIndexMap.size();
    }
    
    public void setSentimentWords(){
        SentimentWords sentiWords = new SentimentWords();
        Set<String> positive= sentiWords.getPositiveWords();
        Set<String> negative = sentiWords.getNegativeWords();
        for (String word : positive){
            if (tokenToIndexMap.containsKey(word)){
                positiveWords.add(tokenToIndexMap.get(word));
            }
        }
        
        for (String word : negative){
            if (tokenToIndexMap.containsKey(word)){
                negativeWords.add(tokenToIndexMap.get(word));
            }
        }
    }
    
    public boolean isPositiveWord(int index){
        return positiveWords.contains(index);
    }
    
    public boolean isNegativeWord(int index){
        return negativeWords.contains(index);
    }
    
}
