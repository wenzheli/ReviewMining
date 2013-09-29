package ml.topicModel.ASUM;

import java.util.HashMap;
import java.util.Map;

import ml.topicModel.common.data.Sentence;


public class LDAModel {
 
    private double alpha;
    private double beta;
    private double gamma;
    
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    private int S;  // # of sentiment labels
    
    private double [][][] theta; // document - topic distributions, size D x S x K
    private double [][] pi;    // document-sentiment distribution
    private double [][][] phi;   // topic-word distributions, size S x K x V
    
    private int[][] z; // latent variable, topic assignment for each sentence
    private int[][] l; // latent variable, sentiment assignments for each sentence
    
    private int [] nDoc;  // nDoc[i] : # of sentences in the document i
    private int [][] nDocSentiment; // nDocSentiment[i][s]: # of sentences assigned to sentiment s, in document i
    private int [][][] nDocSentimentTopic;   // nDocSentimentTopic[i][s][k]: # sentences that
                                             // that are assigned to sentiment s, and topic k
    private int [][] nSentimentTopicWords;  // nSentimentTopicWords[s][k]: # of words assigned to sentiment
                                            // s, and topic k
    private int [][][] nSentimentTopicWordWords; //nSentimentTopicWordWords[s][k][w]: # of word w that
                                                 // are assigned to sentiment s, and topic k. 
    private DataSet dataset;
     
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.alpha = options.alpha;
        this.beta = options.beta;
        this.gamma = options.gamma;
        this.K = options.K;
        this.S = options.S;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][S][K];
        phi = new double[S][K][V];
        pi = new double[D][S];
        
        // initialize temporary variables
        nDoc = new int[D];
        nDocSentiment = new int[D][S];
        nDocSentimentTopic = new int[D][S][K];
        nSentimentTopicWords = new int[S][K];
        nSentimentTopicWordWords = new int[S][K][V];
        
        // initialize latent variable - z and l
        z = new int[D][];
        l = new int[D][];
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            z[i] = new int[numSentences];
            l[i] = new int[numSentences];
            for (int j = 0; j < numSentences; j++){
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() * S);
                z[i][j] = randTopic;
                l[i][j] = randSentiment;
                
                Sentence sentence = d.getSentence(j);
                for (Integer tokenIdx : sentence.getTokens()){
                    // initialize temporary variables
                    nSentimentTopicWords[randSentiment][randTopic]++;
                    nSentimentTopicWordWords[randSentiment][randTopic][tokenIdx]++;
                }
                nDoc[i]++;
                nDocSentiment[i][randSentiment]++;
                nDocSentimentTopic[i][randSentiment][randTopic]++;    
            }
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfSentences(); j++){
                // random sample z[i][j] 
                LatentVariable latentVariable = sample(i,j); // passing sentence j in document i. 
                z[i][j] = latentVariable.getTopic();
                l[i][j] = latentVariable.getSentiment();
            }
        }
    }
    
    // sample new sentence p(z_{i}=k, l_{i}=s | *)
    private LatentVariable sample(int i,  int j){
        Document d = dataset.getDocument(i);
        Sentence sentence = d.getSentence(j);
        
        int oldTopic = z[i][j];
        int oldSentiment = l[i][j];
        
        nDoc[i]--;
        nDocSentiment[i][oldSentiment]--;
        nDocSentimentTopic[i][oldSentiment][oldTopic]--;   
            
        for (Integer tokenIdx : sentence.getTokens()){
            nSentimentTopicWords[oldSentiment][oldTopic]--;
            nSentimentTopicWordWords[oldSentiment][oldTopic][tokenIdx]--;
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
        double[][] p = new double[S][K];
        for (int s = 0; s < S; s++){
            for (int k = 0; k < K; k++){
                
                double devidend[] = new double[sentence.getTokens().size()];
                for (int itr = 0; itr < sentence.getTokens().size(); itr++){
                    devidend[itr] = (beta * V + nSentimentTopicWords[s][k] + itr);
                }
                
                double term = 1;
                int count = 0;
                for (Integer tokenIdx : termCountMap.keySet()){
                    int cnt = termCountMap.get(tokenIdx);
                    for (int itr = 0; itr < cnt; itr++){
                        term = term *  (beta + nSentimentTopicWordWords[s][k][tokenIdx] + itr)/(devidend[count++]);
                    }
                }
                
                p[s][k] = ((alpha + nDocSentimentTopic[i][s][k])/(K*alpha + nDocSentiment[i][s])) 
                        * ((gamma+nDocSentiment[i][s])/(gamma*S+nDoc[i]))
                        * term;
            } 
        }
        
        
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSample(p);
        int newTopic = latentVariable.getTopic();
        int newSentiment = latentVariable.getSentiment();
         
        nDoc[i]++;
        nDocSentiment[i][newSentiment]++;
        nDocSentimentTopic[i][newSentiment][newTopic]++;   
        
        for (Integer tokenIdx : sentence.getTokens()){
            nSentimentTopicWords[newSentiment][newTopic]++;
            nSentimentTopicWordWords[newSentiment][newTopic][tokenIdx]++;
        }
        
        return latentVariable;
    }
    
    public void updateParamters(){
        // update theta
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                for (int k = 0; k < K; k++){
                    theta[i][s][k] = (alpha + nDocSentimentTopic[i][s][k]) / (K * alpha + nDocSentiment[i][s]);
                } 
            }
            
        }
        
        // update phi
        for (int s = 0; s < S; s++){
            for (int k = 0; k < K; k++){
                for (int v = 0; v < V; v++){
                    phi[s][k][v] = (beta + nSentimentTopicWordWords[s][k][v]) / (V * beta + nSentimentTopicWords[s][k]); 
                }
            } 
        }
        
        // update pi
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                pi[i][s] = (gamma + nDocSentiment[i][s])/(gamma * S + nDoc[i]);
            }
        }
        
    }
    
    public double[][][] getTopicDistribution(){
        return theta;
    }
    
    public double[][][] getTopicWordDistribution(){
        return phi;
    }
    
    public double[][] getSentimentDistribution(){
        return pi;
    }
}
