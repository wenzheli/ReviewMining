package ml.topicModel.NGramSentiment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;




public class NGramSentimentModel {
 
    private double alpha;
    private double beta;
    private double gamma;
    private double delta;
    private double omega;
    
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    private int S;  // # of sentiments
    private int I = 2;
    
    private double [][] theta; // document - topic distributions, size D x K
    private double [][] phi;   // topic-word distributions, size K x V
    private double [][] psi;   // topic-word indicator distribution, size K * (V+1) * 2
    private double [][] sigma; // topic-word-word distribution, K * (V+1) * V
    private double [][] pi;    // document-sentiment distribution, D*S
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    private int[][] x; // latent variable, indicator variable for denoting if the current term form bigram with the previous word
    private int[][] l; // latent variable, sentiment assignents for each word D * document.size()
    
    private int [][][] nSentimentTopicWords; // nSentimentTopicWords[s][k][w]: # of instances of word/term w assigned to topic k, and sentiment s, size S*K*V
    private int [][][] nDocSentimentTopic;   // nDocSentimentTopic[i][j]: # of words in document i that assigned to topic j, size D x K
    private int [][] nWordSentimentTopic; // nWordTopic[s][k]: total number of words assigned to topic j, size K. 
    private int [] nWordsSum;     // nWordsSum[i]: total number of words in document i, size D
    private int [][] nDocSentiment; // nDocSentiment[d][s]
    private byte [][][][] nSentimentTopicPrevWordWord; //nTopicWordWord[i][j][k]:  total number of word/term k, assigned to topic i, on the condition that the previous
                                       // previous term is j. size: K * (V+1) * V
    private int [][][] nSentimentTopicPreWord;  // nTopicPreWord[i][j]:  total number of word/term that assigned to topic i, where previous word is j. 
                                     // size: K *(V+1)
    private int [][][][] nPreSentimentPreTopicPreWordIndicator;  // nTopicPreWordIndicator[i][j][k]: total number of indicator variable k, for the condition where
                                                 // the topic of previous word is i, and the previous term/word is j. 
    private int [][][] nPreSentimentPreTopicPreWord;  // size: K * (V+1) 
    
    private DataSet dataset;
    
    Map<List<Integer>, Integer> nGramToIndexMap;
    Map<Integer,List<Integer>> indexToNGramMap;
    
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.alpha = options.alpha;
        this.beta = options.beta;
        this.gamma = options.gamma;
        this.delta = options.delta;
        this.omega = options.omega;
        this.S = options.S;
        this.K = options.K;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[K][V];
        pi = new double[D][S];
        
        // initialize temporary variables
        nSentimentTopicWords = new int[S][K][V];
        nDocSentimentTopic = new int[S][D][K];
        nWordSentimentTopic = new int[S][K];
        nWordsSum = new int[D];
        nDocSentiment = new int[D][S];
        
        nSentimentTopicPrevWordWord = new byte[S][K][V+1][V];
        nSentimentTopicPreWord = new int[S][K][V+1];
        nPreSentimentPreTopicPreWordIndicator = new int[S+1][K+1][V+1][2];
        nPreSentimentPreTopicPreWord = new int[S+1][K+1][V+1];
        
        // initialize latent variable - z and x
        z = new int[D][];
        x = new int[D][];
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            x[i] = new int[numTerms];
            l[i] = new int[numTerms];
            // for each word, randomly assign topic and indicator value. 
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() * S);
                int randIndicatorValue = 0;
                // if the term is starter word, it shouldn't be combine with the previous word. 
                if (j == 0)
                    randIndicatorValue = 0;
                else
                    randIndicatorValue = (int)(Math.random() * I);
                
                z[i][j] = randTopic;
                x[i][j] = randIndicatorValue;
                l[i][j] = randSentiment;
                
                nWordsSum[i]++;
                nDocSentiment[i][randSentiment]++;
                nDocSentimentTopic[i][randSentiment][randTopic]++;
                
                // if starting word....
                if (j == 0){
                    nPreSentimentPreTopicPreWordIndicator[S][K][V][randIndicatorValue]++;
                    nPreSentimentPreTopicPreWord[S][K][V]++;
                } else{
                    nPreSentimentPreTopicPreWordIndicator[l[i][j-1]][z[i][j-1]][d.getToken(j-1)][randIndicatorValue]++;
                    nPreSentimentPreTopicPreWord[l[i][j-1]][z[i][j-1]][d.getToken(j-1)]++;
                }
                
                // use uni-gram 
                if (x[i][j] == 0){
                    nSentimentTopicWords[randSentiment][randTopic][d.getToken(j)]++;  
                    nWordSentimentTopic[randSentiment][randTopic]++;
                    
                } else{ // use bigram 
                    if (j == 0){ // if beginning term
                        nSentimentTopicPrevWordWord[randSentiment][randTopic][V][d.getToken(j)]++;
                        nSentimentTopicPreWord[randSentiment][randTopic][V]++;    
                    } else{ 
                        nSentimentTopicPrevWordWord[randSentiment][randTopic][d.getToken(j-1)][d.getToken(j)]++;
                        nSentimentTopicPreWord[randSentiment][randTopic][d.getToken(j-1)]++;
                    }
                }
                  
            }
            
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            for (int j = d.getNumOfTokens()-1; j >=0; j--){
                // random sample z[i][j] 
                LatentVariable latentVariable = sample(i,j);
                z[i][j] = latentVariable.getTopic();
                x[i][j] = latentVariable.getIndicator();
            }
        }
    }
    
    private LatentVariable sample(int i,  int j){
        Document d = dataset.getDocument(i);
        int termCnt = d.getNumOfTokens();
        int oldTopic = z[i][j];
        int oldSentiment = l[i][j];
        int oldIndicatorValue = x[i][j];
        
        nWordsSum[i]--;  
        nDocSentiment[i][oldSentiment]--;
        nDocSentimentTopic[i][oldSentiment][oldTopic]--;
        
        // update nTopicPreWordIndicator and nTopicPreWordIndicatorSum
        if (j == 0){
            nPreSentimentPreTopicPreWordIndicator[S][K][V][oldIndicatorValue]--;
            nPreSentimentPreTopicPreWord[S][K][V]--;
          
            // and not the last word
            if (j < termCnt - 1){
                nPreSentimentPreTopicPreWordIndicator[oldSentiment][oldTopic][d.getToken(j)][x[i][j+1]]--;
                nPreSentimentPreTopicPreWord[oldSentiment][oldTopic][d.getToken(j)]--;
               
            }
        } else {
            nPreSentimentPreTopicPreWordIndicator[l[i][j-1]][z[i][j-1]][d.getToken(j-1)][oldIndicatorValue]--;
            nPreSentimentPreTopicPreWord[l[i][j-1]][z[i][j-1]][d.getToken(j-1)]--;

            if (j < termCnt - 1){
                nPreSentimentPreTopicPreWordIndicator[oldSentiment][oldTopic][d.getToken(j)][x[i][j+1]]--;
                nPreSentimentPreTopicPreWord[oldSentiment][oldTopic][d.getToken(j)]--;
                
            }
        }
        
        // if unigram...
        if (oldIndicatorValue == 0){
            nSentimentTopicWords[oldSentiment][oldTopic][d.getToken(j)]--;  
            nWordSentimentTopic[oldSentiment][oldTopic]--;
        } else{ // if bigram
            if (j == 0){ // if beginning term
                nSentimentTopicPrevWordWord[oldSentiment][oldTopic][V][d.getToken(j)]--;
                nSentimentTopicPreWord[oldSentiment][oldTopic][V]--; 
                
            } else{ 
                nSentimentTopicPrevWordWord[oldSentiment][oldTopic][d.getToken(j-1)][d.getToken(j)]--;
                nSentimentTopicPreWord[oldSentiment][oldTopic][d.getToken(j-1)]--;
            }
        }
        
      
        
        // compute p(z[i][j]|*)
        double[][][] p = new double[S][K][I];
        for (int v = 0; v < S; v++){
            for (int k = 0; k < K; k++){
                for (int s = 0; s < I; s++){
                    // if uni-gram
                    if (s == 0){
                        if (j == 0){
                            
                            p[v][k][s] = ((alpha + nDocSentimentTopic[i][v][k])/(K*alpha + nDocSentiment[i][v])) 
                                    * ((omega + nDocSentiment[i][v])/(omega + nWordsSum[i]))
                                    * ((beta+nSentimentTopicWords[v][k][d.getToken(j)])/(V*beta+nWordSentimentTopic[v][k]))
                                    * ((gamma + nPreSentimentPreTopicPreWordIndicator[S][K][V][s])/(I*gamma + nPreSentimentPreTopicPreWord[S][K][V]));
                            
                        } else{
                            p[v][k][s] = ((alpha + nDocSentimentTopic[i][v][k])/(K*alpha + nDocSentiment[i][v])) 
                                    * ((omega + nDocSentiment[i][v])/(omega + nWordsSum[i]))
                                    * ((beta+nSentimentTopicWords[v][k][d.getToken(j)])/(V*beta+nWordSentimentTopic[v][k]))
                                    * ((gamma + nPreSentimentPreTopicPreWordIndicator[l[i][j-1]][(z[i][j-1])][d.getToken(j-1)][s])/(I*gamma + nPreSentimentPreTopicPreWord[l[i][j-1]][z[i][j-1]][d.getToken(j-1)]));
                           
                        }
                        
                    } else{
                        if (j == 0){
                            p[v][k][s] = ((alpha + nDocSentimentTopic[i][v][k])/(K*alpha + nDocSentiment[i][v]))  
                                    * ((omega + nDocSentiment[i][v])/(omega + nWordsSum[i]))
                                    * ((delta+nSentimentTopicPrevWordWord[S][k][V][d.getToken(j)])/(V*delta+nSentimentTopicPreWord[S][k][V]))
                                    * ((gamma + nPreSentimentPreTopicPreWordIndicator[S][K][V][s])/(I*gamma + nPreSentimentPreTopicPreWord[S][K][V]));
                            
                        }else {
                            p[v][k][s] = ((alpha + nDocSentimentTopic[i][v][k])/(K*alpha + nDocSentiment[i][v])) 
                                    * ((omega + nDocSentiment[i][v])/(omega + nWordsSum[i]))
                                    * ((delta+nSentimentTopicPrevWordWord[v][k][d.getToken(j-1)][d.getToken(j)])/(V*delta+nSentimentTopicPreWord[v][k][d.getToken(j-1)]))
                                    * ((gamma + nPreSentimentPreTopicPreWordIndicator[l[i][j-1]][(z[i][j-1])][d.getToken(j-1)][s])/(I*gamma + nPreSentimentPreTopicPreWord[l[i][j-1]][z[i][j-1]][d.getToken(j-1)]));
                           
                        }
                    }
                    
                }
                
            }
        }
        
        
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSample(p);
        int newTopic = latentVariable.getTopic();
        int newIndicatorValue = latentVariable.getIndicator();
        int newSentiment = latentVariable.getSentiment();
        
        nWordsSum[i]++;      
        nDocSentiment[i][newSentiment]++;
        nDocSentimentTopic[i][newSentiment][newTopic]++;
        
        // update nTopicPreWordIndicator and nTopicPreWordIndicatorSum
        if (j == 0){
            nPreSentimentPreTopicPreWordIndicator[S][K][V][newIndicatorValue]++;
            nPreSentimentPreTopicPreWord[S][K][V]++;
            // and not the last word
            if (j < termCnt - 1){
                nPreSentimentPreTopicPreWordIndicator[newSentiment][newTopic][d.getToken(j)][x[i][j+1]]++;
                nPreSentimentPreTopicPreWord[newSentiment][newTopic][d.getToken(j)]++;
            }
        } else {
            nPreSentimentPreTopicPreWordIndicator[l[i][j-1]][z[i][j-1]][d.getToken(j-1)][newIndicatorValue]++;
            nPreSentimentPreTopicPreWord[l[i][j-1]][z[i][j-1]][d.getToken(j-1)]++;
            if (j < termCnt - 1){
                nPreSentimentPreTopicPreWordIndicator[newSentiment][newTopic][d.getToken(j)][x[i][j+1]]++;
                nPreSentimentPreTopicPreWord[newSentiment][newTopic][d.getToken(j)]++;
            }
        }
        
        // if unigram...
        if (newIndicatorValue == 0){
            nSentimentTopicWords[newSentiment][newTopic][d.getToken(j)]++;  
            nWordSentimentTopic[newSentiment][newTopic]++;
        } else{ // if bigram
            if (j == 0){ // if beginning term
                nSentimentTopicPrevWordWord[newSentiment][newTopic][V][d.getToken(j)]++;
                nSentimentTopicPreWord[newSentiment][newTopic][V]++;            
            } else{ 
                nSentimentTopicPrevWordWord[newSentiment][newTopic][d.getToken(j-1)][d.getToken(j)]++;
                nSentimentTopicPreWord[newSentiment][newTopic][d.getToken(j-1)]++;
            }
        }
        
        return latentVariable;
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
                phi[k][v] = (beta + nTopicWords[k][v]) / (V * beta + nWordTopic[k]); 
            }
        }
    }
    
    public double[][] getTopWordsFromNGram(){
        List<NGramDocument> nGramDocuments = new ArrayList<NGramDocument>();
        
        for (int i = 0; i < D; i++){
            List<List<Integer>> terms = new ArrayList<List<Integer>>();
            List<Integer> topics = new ArrayList<Integer>();
            List<Integer> words;
            Document d = dataset.getDocument(i);
            for (int j = 1; j < d.getNumOfTokens(); j++){
                if(x[i][j] == 1){
                    words = new ArrayList<Integer>();
                    words.add(d.getToken(j-1));
                    j++;
                    while (j< d.getNumOfTokens() && x[i][j] == 1){
                        words.add(d.getToken(j-1));
                        j++;
                    }
                    // add last word. 
                    words.add(d.getToken(j-1));
                    topics.add(z[i][j-1]);
                    terms.add(words);
                }   
            }
            
            NGramDocument doc = new NGramDocument();
            doc.setTerms(terms);
            doc.setTopics(topics);
            nGramDocuments.add(doc);
        }
        
        int index = 0;
        nGramToIndexMap = new HashMap<List<Integer>, Integer>();
        indexToNGramMap = new HashMap<Integer, List<Integer>>();
        for (NGramDocument nGramDoc : nGramDocuments){
            for (List<Integer> term : nGramDoc.getTerms()){
                if (!nGramToIndexMap.containsKey(term)){
                    nGramToIndexMap.put(term, index);
                    indexToNGramMap.put(index, term);
                    index++;
                }
            }
        }
        
        double[][] topicNGramDist = new double[K][nGramToIndexMap.keySet().size()];
        for (NGramDocument nGramDoc : nGramDocuments){
            List<List<Integer>> nGramTerms = nGramDoc.getTerms();
            List<Integer> nGramTopics = nGramDoc.getTopics();
            for (int i = 0; i < nGramTerms.size(); i++){
                int idx = nGramToIndexMap.get(nGramTerms.get(i));
                topicNGramDist[nGramTopics.get(i)][idx]++;
            }
        }
        
        return topicNGramDist;
             
    }
    
    public Map<List<Integer>, Integer> getNGramToIndexMap(){
        return nGramToIndexMap;
    }
    
    public Map<Integer, List<Integer>> getIndexToNGramMap(){
        return indexToNGramMap;
    }
    
    
    public double[][] getTopicDistribution(){
        return theta;
    }
    
    public double[][] getTopicWordDistribution(){
        return phi;
    }
}
