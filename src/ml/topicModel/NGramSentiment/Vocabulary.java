package ml.topicModel.NGramSentiment;

import java.util.HashMap;
import java.util.Map;

public class Vocabulary {
    
    Map<String, Integer> tokenToIndexMap = new HashMap<String, Integer>();
    Map<Integer, String> indexTotokenMap = new HashMap<Integer, String>();
        
    public void settokenToIndex(Map<String, Integer> tokenToIndexMap){
        this.tokenToIndexMap = tokenToIndexMap;
    }
    
    public void setIndexTotokenMap(Map<Integer, String> indexTotokenMap){
        this.indexTotokenMap = indexTotokenMap;
    }
    
    public int getVocabularySize(){
        return tokenToIndexMap.size();
    }
    
}
