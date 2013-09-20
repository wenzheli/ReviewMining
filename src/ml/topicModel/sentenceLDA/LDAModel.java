package ml.topicModel.sentenceLDA;

import ml.topicModel.lda.LatentVariable;


public class LDAModel {
 
    private double alpha;
    private double beta;
    private double gamma;
    
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
        this.K = options.K;
        this.S = options.S;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[S][K][V];
        
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
            Document d = dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            l[i] = new int[numTerms];
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() * S);
                z[i][j] = randTopic;
                l[i][j] = randSentiment;
                
                nSentimentTopicWords[randSentiment][randTopic]++;
                nSentimentTopicWordWords[randSentiment][randTopic][d.getToken(j)]++;
                nDocSentimentTopicWords[i][randSentiment][randTopic]++;
                nDocSentimentWords[i][randSentiment]++;
                nDocWords[i]++;
                
            }
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                // random sample z[i][j] 
                LatentVariable latentVariable = sample(i,j);
                z[i][j] = latentVariable.getTopic();
                l[i][j] = latentVariable.getSentiment();
            }
        }
    }
    
    private int sampleNewTopic(int i,  int j){
        Document d = dataset.getDocument(i);
        int oldTopic = z[i][j];
        nTopicWords[oldTopic][d.getToken(j)]--;
        nDocTopic[i][oldTopic]--;
        nWordTopicSum[oldTopic]--;
        nWordsSum[i]--;
        
        // compute p(z[i][j]|*)
        double[] p = new double[K];
        for (int k = 0; k < K; k++){
            p[k] = ((alpha + nDocTopic[i][k])/(K*alpha + nWordsSum[i])) 
                    * ((beta+nTopicWords[k][d.getToken(j)])/(V*beta+nWordTopicSum[k]));
        }
        
        // sample the topic topic from the distribution p[j].
        int newTopic = DistributionUtils.getSample(p);
        
        nTopicWords[newTopic][d.getToken(j)]++;
        nDocTopic[i][newTopic]++;
        nWordTopicSum[newTopic]++;
        nWordsSum[i]++;
        
        return newTopic;
    }
    
    public void updateParamters(){
        // update theta
        for (int i = 0; i < D; i++){
            for (int k = 0; k < K; k++){
                theta[i][k] = (alpha + nDocTopic[i][k]) / (K * alpha + nWordsSum[i]);
            }
        }
        
        // update phi
        for (int k = 0; k < K; k++){
            for (int v = 0; v < V; v++){
                phi[k][v] = (beta + nTopicWords[k][v]) / (V * beta + nWordTopicSum[k]); 
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
