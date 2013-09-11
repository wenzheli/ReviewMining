package ml.topicModel.lda;

import java.util.List;
import java.util.Map;



public class DataSet {
    public class Document{
        int nUniqueWords;   // # of unique words
        Map<Integer, Integer> wordsCounts; // occurance of each word
        
        public int getNumUniqueWords(){
            return nUniqueWords;
        }
        
        public Map<Integer, Integer> getWordsCounts(){
            return wordsCounts;
        }
    }
    
    List<Document> documents;
    int numOfDocuments;
    Vocabulary vocab;
    
    // read the correctly formatted data set from file. No need for preprocessing.  
    public void loadFormattedDataSet(String fileName){
        
    }
    
    // need preprocessing:  stop words + stemming 
    public void loadDataset(String fileName){
        
    }
}
