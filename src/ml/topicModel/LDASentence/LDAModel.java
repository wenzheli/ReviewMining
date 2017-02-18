package ml.topicModel.LDASentence;

import java.util.HashMap;
import java.util.Map;

import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.data.SDocument;
import ml.topicModel.common.data.Sentence;
import ml.topicModel.utils.DistributionUtils;

public class LDAModel { 
    private double alpha;
    private double beta;
   
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    
    private double [][] theta; // document - topic distributions, size D x S x K
    private double [][] phi;   // topic-word distributions, size S x K x V
    
    private int[][] z; // latent variable, topic assignment for each sentence
    
    private int [] nDoc;  // nDoc[i] : # of sentences in the document i
    private int [][] nDocTopic;   // nDocSentimentTopic[i][k]: # sentences that
                                             // that are assigned to sentiment s, and topic k
    private int [] nTopicWords;  // nSentimentTopicWords[k]: # of words assigned to 
                                            //  and topic k
    private int [][] nTopicWordWords; //nSentimentTopicWordWords[k][w]: # of word w that
                                                 // are assigned to sentiment s, and topic k. 
    private DataSet dataset;
     
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.alpha = options.alpha; 
        this.K = options.K;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        this.dataset = dataset;
           
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[K][V];
       
        // initialize temporary variables
        nDoc = new int[D];
        nDocTopic = new int[D][K];
        nTopicWords = new int[K];
        nTopicWordWords = new int[K][V];
        
        // initialize latent variable - z and l
        z = new int[D][];
        for (int i = 0; i < D; i++){
            SDocument d = (SDocument) dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            z[i] = new int[numSentences];
            for (int j = 0; j < numSentences; j++){
                int randTopic = (int)(Math.random() * K);
                z[i][j] = randTopic;  
                
                Sentence sentence = d.getSentence(j);
                for (Integer tokenIdx : sentence.getTokens()){
                    // initialize temporary variables
                    nTopicWords[randTopic]++;
                    nTopicWordWords[randTopic][tokenIdx]++;
                }
                nDoc[i]++;
                nDocTopic[i][randTopic]++;    
            }
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            SDocument d = (SDocument) dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfSentences(); j++){
                // random sample z[i][j] 
                int newTopic = sample(i,j); // passing sentence j in document i. 
                z[i][j] = newTopic;
            }
        }
    }
    
    // sample new sentence p(z_{i}=k, l_{i}=s | *)
    private int sample(int i,  int j){
        SDocument d = (SDocument) dataset.getDocument(i);
        Sentence sentence = d.getSentence(j);
        
        int oldTopic = z[i][j];
       
        nDoc[i]--;
        nDocTopic[i][oldTopic]--;   
            
        for (Integer tokenIdx : sentence.getTokens()){
            nTopicWords[oldTopic]--;
            nTopicWordWords[oldTopic][tokenIdx]--;
        }
          
        Map<Integer, Integer> termCountMap = new HashMap<Integer, Integer>();
        for (Integer tokenIdx : sentence.getTokens()){
            if (!termCountMap.containsKey(tokenIdx)){
                termCountMap.put(tokenIdx, 1);
            } else{
                int cnt = termCountMap.get(tokenIdx);
                termCountMap.put(tokenIdx, cnt + 1);
            }
        }
       
        // compute p(z[i][j]|*)
        double[] p = new double[K];
        for (int k = 0; k < K; k++){       
            double devidend[] = new double[sentence.getTokens().size()];
            for (int itr = 0; itr < sentence.getTokens().size(); itr++){
                devidend[itr] = (beta * V + nTopicWords[k] + itr);
            }
            
            double term = 1;
            int count = 0;
            for (Integer tokenIdx : termCountMap.keySet()){
                int cnt = termCountMap.get(tokenIdx);
                for (int itr = 0; itr < cnt; itr++){
                    term = term *  (beta + nTopicWordWords[k][tokenIdx] + itr)/(devidend[count++]);
                }
            }
            
            p[k] = ((alpha + nDocTopic[i][k])/(K*alpha + nDoc[i])) 
                    * term;
        } 
            
        // sample the topic topic from the distribution p[j].
        int newTopic = DistributionUtils.getSample(p);
         
        nDoc[i]++;
        nDocTopic[i][newTopic]++;   
        
        for (Integer tokenIdx : sentence.getTokens()){
            nTopicWords[newTopic]++;
            nTopicWordWords[newTopic][tokenIdx]++;
        }    
        return newTopic;
    }
    
    public void updateParamters(){
        // update theta
        for (int i = 0; i < D; i++){
            for (int k = 0; k < K; k++){
                theta[i][k] = (alpha + nDocTopic[i][k]) / (K * alpha + nDoc[i]);
            }         
        }  
        
        // update phi
        for (int k = 0; k < K; k++){
            for (int v = 0; v < V; v++){
                phi[k][v] = (beta + nTopicWordWords[k][v]) / (beta *V + nTopicWords[k]); 
            }
        } 
    }
    
    public double[][] getTopicDistribution(){
        return theta;
    }
    
    public double[][] getTopicWordDistribution(){
        return phi;
    }
}
