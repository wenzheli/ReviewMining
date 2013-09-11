package ml.topicModel.lda;

public class LDAModel {
    
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    
    private double [][] theta; // document - topic distributions, size M x K
    private double [][] phi;   // topic-word distributions, size K x V
    
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    
    private int [][] nWordTopic; // nWordTopic[i][j]: # of instances of word/term i assigned to topic j, size V x K
    private int [][] nDocTopic; // nDocTopic[i][j]: # of words in document i that assigned to topic j, size D x K
    private int [] nWordTopicSum; // nWordTopicSum[j]: total number of words assigned to topic j, size K. 
    private int [] nWordsSum; // nWordsSum[i]: total number of words in document i, size D
    
    // temp variables for sampling
    private double [] p; 
    
    
    public void saveParamtersToFile(){
        
    }
    
    public void loadParametersFromFile(){
        
    }
    
    public void printTopWords(){
        
    }
    
    public void saveTopWords(){
        
    }
    
    
}
