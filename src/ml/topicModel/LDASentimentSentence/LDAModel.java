package ml.topicModel.LDASentimentSentence;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.data.LatentVariable;
import ml.topicModel.common.data.SDocument;
import ml.topicModel.common.data.Sentence;
import ml.topicModel.common.data.Vocabulary;
import ml.topicModel.utils.DistributionUtils;

public class LDAModel { 
    private double alpha;
    private double[][] beta;
    private double[] betaSum;
    private double[] gamma;
    
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
        this.K = options.K;
        this.S = options.S;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        this.dataset = dataset;
        this.betaSum = new double[S];
        this.gamma = new double[S];
        // initialize gamma
        gamma[0] = 1; // for positive
        gamma[1] = 1; // for negative
        
        // initialzie beta using asymmetric prior
        this.beta = new double[S][V];
        for (int i = 0; i < beta.length; i++){
            Arrays.fill(beta[i], 0.001);
        }
        
        Vocabulary vocab = dataset.getVocabulary();
        Set<Integer> positiveWords = vocab.getPositiveWordS();
        for (Integer tokenId : positiveWords){
            beta[1][tokenId] = 0;
        }
        Set<Integer> negativeWords = vocab.getNegativeWords();
        for (Integer tokenId : negativeWords){
            beta[0][tokenId] = 0;
        }
        
        for (double value : beta[0]){
            betaSum[0]+=value;
        }
        for (double value : beta[1]){
            betaSum[1]+=value;
        }
       
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
            SDocument d = (SDocument) dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            z[i] = new int[numSentences];
            l[i] = new int[numSentences];
            for (int j = 0; j < numSentences; j++){
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() * S);
                 
                Sentence sentence = d.getSentence(j);
                for (Integer tokenIdx : sentence.getTokens()){
                   if (vocab.positiveWords.contains(tokenIdx)){
                       randSentiment = 0;
                   } else if(vocab.negativeWords.contains(tokenIdx)){
                       randSentiment  = 1;
                   }
                }
                
                z[i][j] = randTopic;  
                l[i][j] = randSentiment;
                
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
            SDocument d = (SDocument) dataset.getDocument(i);
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
        SDocument d = (SDocument) dataset.getDocument(i);
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
                    devidend[itr] = (betaSum[s] + nSentimentTopicWords[s][k] + itr);
                }
                
                double term = 1;
                int count = 0;
                for (Integer tokenIdx : termCountMap.keySet()){
                    int cnt = termCountMap.get(tokenIdx);
                    for (int itr = 0; itr < cnt; itr++){
                        term = term *  (beta[s][tokenIdx] + nSentimentTopicWordWords[s][k][tokenIdx] + itr)/(devidend[count++]);
                    }
                }
                
                p[s][k] = ((alpha + nDocSentimentTopic[i][s][k])/(K*alpha + nDocSentiment[i][s])) 
                        * ((gamma[s]+nDocSentiment[i][s])/(gamma[0]+gamma[1]+nDoc[i]))
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
                    phi[s][k][v] = (beta[s][v] + nSentimentTopicWordWords[s][k][v]) / (betaSum[s] + nSentimentTopicWords[s][k]); 
                }
            } 
        }
        
        // update pi
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                pi[i][s] = (gamma[s] + nDocSentiment[i][s])/(gamma[0]+gamma[1] + nDoc[i]);
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
