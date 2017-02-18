package ml.topicModel.common.data;

import java.util.List;

/**
 * Representation of documents. 
 * @author wenzhe
 *
 */
public class DataSet {
    public String dataSetName = "yelp";
    List<Document> documents;
    int numDocuments;
    public Vocabulary vocab;
   
    public List<Document> getDocuments(){
        return documents;
    }
    
    public int getDocumentCount(){
        return documents.size();
    }
    
    public Vocabulary getVocabulary(){
        return vocab;
    }
    
    public Document getDocument(int index){
        return documents.get(index);
    }
    
    public void setDocuments(List<Document> documents){
        this.documents = documents;
    }
    
    public void setNumDocuments(int numDocuments){
        this.numDocuments = numDocuments;
    }
    
    public void setVocabulary(Vocabulary vocab){
        this.vocab = vocab;
    }
}
