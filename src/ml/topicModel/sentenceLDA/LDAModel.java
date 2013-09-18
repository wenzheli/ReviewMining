package ml.topicModel.sentenceLDA;


public class LDAModel {
 
    private double alpha;
    private double beta;
    
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    
    private double [][] theta; // document - topic distributions, size D x K
    private double [][] phi;   // topic-word distributions, size K x V
    
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    
    private int [][] nTopicWords; // nTopicWords[i][j]: # of instances of word/term j assigned to topic i, size K*V
    private int [][] nDocTopic;   // nDocTopic[i][j]: # of words in document i that assigned to topic j, size D x K
    private int [] nWordTopicSum; // nWordTopicSum[j]: total number of words assigned to topic j, size K. 
    private int [] nWordsSum;     // nWordsSum[i]: total number of words in document i, size D
    
    private DataSet dataset;
     
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.alpha = options.alpha;
        this.beta = options.beta;
        this.K = options.K;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[K][V];
        
        // initialize temporary variables
        nTopicWords = new int[K][V];
        nDocTopic = new int[D][K];
        nWordTopicSum = new int[K];
        nWordsSum = new int[D];
        
        // initialize latent variable - z
        z = new int[D][];
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                z[i][j] = randTopic;
                nTopicWords[randTopic][d.getToken(j)]++;
                nDocTopic[i][randTopic]++;
                nWordTopicSum[randTopic]++;      
            }
            nWordsSum[i] = numTerms;
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                // random sample z[i][j] 
                int newTopic = sampleNewTopic(i,j);
                z[i][j] = newTopic;
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
