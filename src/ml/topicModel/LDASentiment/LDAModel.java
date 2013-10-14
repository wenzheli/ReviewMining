package ml.topicModel.LDASentiment;

import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.data.LatentVariable;
import ml.topicModel.common.data.Vocabulary;
import ml.topicModel.common.data.WDocument;
import ml.topicModel.utils.DistributionUtils;

public class LDAModel {
 
    private double alpha;
    private double beta;
    private double[] gamma;
    
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int S = 2; // # of sentiments
    private int V;  // vocabulary size
    
    private double [][] theta; // document - topic distributions, size D x K
    private double [][][] phi;   // sentiment-topic-word distributions, size S * K * V
    private double [][] pi; // document - sentiment distribution, D * S
    
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    private int[][] l; // latent variable, sentiment assignment for each word. D * document.size()
    
    private int [][] nSentimentTopicWords; // nSentimentTopicWords[i][j]:  # of words assigned to sentiment i, and topic j. 
    private int [][][] nSentimentTopicWordWords; // nSentimentTopicWordWords[i][j][k]: # of word w_{k} assigned to sentiment i, topic j, 
    
    private int [][][] nDocSentimentTopicWords; // total number of words in document i, assigned to sentiment j, and topic k. 
    private int [][] nDocSentimentWords; // nDocSentimentWords[i][j]: total number of words in document i, assigned to sentiment j
    
    private int [] nDocWords;     // nWordsSum[i]: total number of words in document i, size D
    
    private DataSet dataset;
     
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.alpha = options.alpha;
        this.beta = options.beta;
        this.gamma = new double[2];
        gamma[0] = 1;
        gamma[1] = 1;
        this.K = options.K;
        this.S = options.S;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[S][K][V];
        pi = new double[D][S];
        
        // initialize temporary variables
        nSentimentTopicWords = new int[S][K];
        nSentimentTopicWordWords = new int[S][K][V];
        nDocSentimentTopicWords = new int[D][S][K];
        nDocSentimentWords = new int[D][S];
        nDocWords = new int[D];
        

        // initialize latent variable - z
        z = new int[D][];
        l = new int[D][];
        for (int i = 0; i < D; i++){
            WDocument d = (WDocument) dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            l[i] = new int[numTerms];
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() * S);
                z[i][j] = randTopic;
                
                Vocabulary vocab = dataset.getVocabulary();
                if (vocab.isPositiveWord(d.getToken(j)))
                    l[i][j] = 0;
                else if (vocab.isNegativeWord(d.getToken(j)))
                    l[i][j] = 1;
                else
                    l[i][j] = randSentiment;
                
                nSentimentTopicWords[l[i][j]][randTopic]++;
                nSentimentTopicWordWords[l[i][j]][randTopic][d.getToken(j)]++;
                nDocSentimentTopicWords[i][l[i][j]][randTopic]++;
                nDocSentimentWords[i][l[i][j]]++;
                nDocWords[i]++;   
            }
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            WDocument d = (WDocument) dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                // random sample z[i][j] 
                LatentVariable latentVariable = sample(i,j);
                z[i][j] = latentVariable.getTopic();
                l[i][j] = latentVariable.getSentiment();
            }
        }
    }
    
    private LatentVariable sample(int i,  int j){
        WDocument d = (WDocument) dataset.getDocument(i);
        int oldTopic = z[i][j];
        int oldSentiment = l[i][j];
        
        nSentimentTopicWords[oldSentiment][oldTopic]--;           
        nSentimentTopicWordWords[oldSentiment][oldTopic][d.getToken(j)]--;      
        nDocSentimentTopicWords[i][oldSentiment][oldTopic]--;      
        nDocSentimentWords[i][oldSentiment]--;       
        nDocWords[i]--;
        
        // compute p(z[i][j]|*)
        double[][] p = new double[S][K];
        for (int s = 0; s < S; s++){
            Vocabulary vocab = dataset.vocab;
            if(vocab.isPositiveWord(d.getToken(j)) && s==1 
                    || vocab.isNegativeWord(d.getToken(j)) && s==0){
                continue;
            }
            
            for (int k = 0; k < K; k++){
                p[s][k] = ((alpha + nDocSentimentTopicWords[i][s][k])/(K*alpha + nDocSentimentWords[i][s])) 
                        * ((beta+nSentimentTopicWordWords[s][k][d.getToken(j)])/(V*beta+nSentimentTopicWords[s][k]))
                        * ((gamma[s] + nDocSentimentWords[i][s])/(gamma[0]+gamma[1] + nDocWords[i]));
            } 
        }
        
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSample(p);
        int newTopic = latentVariable.getTopic();
        int newSentiment = latentVariable.getSentiment();
        
        nSentimentTopicWords[newSentiment][newTopic]++;
        nSentimentTopicWordWords[newSentiment][newTopic][d.getToken(j)]++;
        nDocSentimentTopicWords[i][newSentiment][newTopic]++;
        nDocSentimentWords[i][newSentiment]++;
        nDocWords[i]++;
        
        return latentVariable;
    }
    
    public void updateParamters(){
       
        // update phi
        for (int s = 0; s < S; s++){
            for (int k = 0; k < K; k++){
                for (int v = 0; v < V; v++){
                    phi[s][k][v] = (beta + nSentimentTopicWordWords[s][k][v]) / (V * beta + nSentimentTopicWords[s][k]); 
                }
            }
        }
        
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                pi[i][s] = ((gamma[s] + nDocSentimentWords[i][s])/(gamma[0] + gamma[1] + nDocWords[i]));
            }
        }
    }
    
    
    public double[][][] getTopicWordDistribution(){
        return phi;
    }
    
    public double[][] getSentimentDistribution(){
        return pi;
    }
}
