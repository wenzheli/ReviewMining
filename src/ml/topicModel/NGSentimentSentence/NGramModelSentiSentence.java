package ml.topicModel.NGSentimentSentence;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;



public class NGramModelSentiSentence {
 
    private double alpha;
    private double beta;
    private double gamma;
    private double delta;
    private double omega;
    
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    private int S;
    private int I = 2;
    
    private double [][] theta; // document - topic distributions, size D x K
    private double [][] phi;   // topic-word distributions, size K x V
    private double [][] psi;   // topic-word indicator distribution, size K * (V+1) * 2
    private double [][] sigma; // topic-word-word distribution, K * (V+1) * V
    
    private int[][] z; // latent variable, topic assignments for each word. D * number of sentences
    private int[][][] x; // latent variable, indicator variable for denoting if the current term form bigram with the previous word
    private int[][] l; // latent variable, indicator variable for sentiment D * number of sentences
    
    // used for topic and sentiments.
    private byte [][][] nDocSentimentTopic;   // D*S*K nDocSentimentTopic[d][s][k]: # of sentences in document d that assigned to topic k, and sentiment s
    private short [][] nDocSentiment; // nDocSentiment[d][s]: # of sentences assigned to sentiment s, in document d. 
    private int [] nTotalSentences;     // nTotalSentences[i]: total number of sentences in document i, size D
    
    private int [][] nWordSentimentTopic; // S * K nWordTopic[s][k]: total number of words assigned to topic j, size K. 
    private int [][][] nWordSentimentTopicWords; // S*K*V nSentimentTopicWords[s][k][w]: # of instances of word/term w assigned to topic k, and sentiment s, 
    
    
    // used for bigram
    private byte [][][][] nSentimentTopicPrevWordWord; //nTopicWordWord[i][j][k]:  total number of word/term k, assigned to topic i, on the condition that the previous
                                       // previous term is j. size: K * (V+1) * V
    private short [][][] nSentimentTopicPreWord;  // nTopicPreWord[i][j]:  total number of word/term that assigned to topic i, where previous word is j. 
                                     // size: K *(V+1)
    // used for indicator variable. 
    private short [][][][] nPreSentimentPreTopicPreWordIndicator;  // nTopicPreWordIndicator[i][j][k]: total number of indicator variable k, for the condition where
                                                 // the topic of previous word is i, and the previous term/word is j. 
    private short [][][] nPreSentimentPreTopicPreWord;  // size: K * (V+1) 
    
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
        
        this.K = options.K;
        this.S = options.S;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[K][V];
        
        // initialize temporary variables
        nDocSentimentTopic = new byte[D][S][K];
        nDocSentiment = new short[D][S];
        nTotalSentences = new int[D];
        nWordSentimentTopic = new int[S][K];
        nWordSentimentTopicWords = new int[S][K][V];
        
        nSentimentTopicPrevWordWord = new byte[S][K][V+1][V];
        nSentimentTopicPreWord = new short[S][K][V+1];
        nPreSentimentPreTopicPreWordIndicator = new short[S][K][V+1][I];
        nPreSentimentPreTopicPreWord = new short[S][K][V+1];
        
        // initialize latent variable - z and x
        // i: document  j: sentence   k: words
        z = new int[D][];
        l = new int[D][];
        x = new int[D][][];
        
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            // assign topic and sentiment to each sentence. 
            z[i] = new int[numSentences];
            l[i] = new int[numSentences];
            x[i] = new int[numSentences][];
            
            // iterate each sentence. 
            for (int j = 0; j < numSentences; j++){
                // assign topic and sentiment for each sentence. 
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() *S);
                z[i][j] = randTopic;
                l[i][j] = randSentiment;
                
                Sentence sentence = d.getSentence(j);
                x[i][j] = new int[sentence.getTokens().size()];
                
                // iterate each token. 
                for (int k = 0; k < sentence.getTokens().size(); k++){
                    int randIndicator = 0;
                    if (k > 0){
                        randIndicator = (int)(Math.random()*I);
                    }
                    x[i][j][k] = randIndicator;
                    
                    // update indicator parameters. 
                    if (k == 0){ // beginning word
                        nPreSentimentPreTopicPreWordIndicator[randSentiment][randTopic][V][randIndicator]++;
                        nPreSentimentPreTopicPreWord[randSentiment][randTopic][V]++;
                    }else{ 
                        nPreSentimentPreTopicPreWordIndicator[randSentiment][randTopic][sentence.getToken(k-1)][randIndicator]++;
                        nPreSentimentPreTopicPreWord[randSentiment][randTopic][sentence.getToken(k-1)]++;
                    }
                    
                    // update word probability. 
                    if (randIndicator == 0){ // use unigram
                        nWordSentimentTopic[randSentiment][randTopic]++;
                        nWordSentimentTopicWords[randSentiment][randTopic][sentence.getToken(k)]++;
                    } else{ // use bigram. 
                        if (k ==0){ // if begining word
                            nSentimentTopicPrevWordWord[randSentiment][randTopic][V][sentence.getToken(k)]++;
                            nSentimentTopicPreWord[randSentiment][randTopic][V]++;
                        }else{
                            nSentimentTopicPrevWordWord[randSentiment][randTopic][sentence.getToken(k-1)][sentence.getToken(k)]++;
                            nSentimentTopicPreWord[randSentiment][randTopic][sentence.getToken(k-1)]++;
                        }
                    }
                    
                }
                
                nDocSentimentTopic[i][randSentiment][randTopic]++;
                nDocSentiment[i][randSentiment]++;
                nTotalSentences[i]++;
            }       
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            for (int j = 0; j < numSentences; j++){
                // random sample z[i][j] 
                LatentVariable latentVariable = sample(i,j);
                z[i][j] = latentVariable.getTopic();
                l[i][j] = latentVariable.getSentiment();
                x[i][j] = latentVariable.getIndicators();
            }
            
            // System.out.println(i);
        }
        
        
    }
    
    // sampling z,l,x for sentence j, in document i. 
    private LatentVariable sample(int i,  int j){
        Document d = dataset.getDocument(i);
        Sentence sentence = d.getSentence(j);
      
        int oldTopic = z[i][j];
        int oldSentiment = l[i][j];
        int[] oldIndicators = x[i][j];
        
        nDocSentimentTopic[i][oldSentiment][oldTopic]--;
        nDocSentiment[i][oldSentiment]--;
        nTotalSentences[i]--;
        
        // iterate each word. 
        for (int k = 0; k < sentence.getTokens().size(); k++){
            int oldIndicator = oldIndicators[k];
            
            // update indicator related variables. 
            if (k == 0){ // beginning word
                nPreSentimentPreTopicPreWordIndicator[oldSentiment][oldTopic][V][oldIndicator]--;
                nPreSentimentPreTopicPreWord[oldSentiment][oldTopic][V]--;
            }else{ 
                nPreSentimentPreTopicPreWordIndicator[oldSentiment][oldTopic][sentence.getToken(k-1)][oldIndicator]--;
                nPreSentimentPreTopicPreWord[oldSentiment][oldTopic][sentence.getToken(k-1)]--;
            }
            
            // update word probability. 
            if (oldIndicator == 0){ // use unigram
                nWordSentimentTopic[oldSentiment][oldTopic]--;
                nWordSentimentTopicWords[oldSentiment][oldTopic][sentence.getToken(k)]--;
            } else{ // use bigram. 
                if (k ==0){ // if begining word
                    nSentimentTopicPrevWordWord[oldSentiment][oldTopic][V][sentence.getToken(k)]--;
                    nSentimentTopicPreWord[oldSentiment][oldTopic][V]--;
                }else{
                    nSentimentTopicPrevWordWord[oldSentiment][oldTopic][sentence.getToken(k-1)][sentence.getToken(k)]--;
                    nSentimentTopicPreWord[oldSentiment][oldTopic][sentence.getToken(k-1)]--;
                }
            }
        }
        
        
        
        /*
         *  compute p(z[i][j]|*)
         */
        int numTokens = sentence.getTokens().size();
        // generate random permutation
        int totalPermutation = (int) Math.pow(2, numTokens-1);
        if (totalPermutation == 0){
            int aaa =1;
        }
        
        double[][][] p = new double[S][K][totalPermutation];
        
        for (int s = 0; s < S; s++){
            for (int k = 0; k < K; k++){
                double term1 = ((alpha + nDocSentimentTopic[i][s][k])/(alpha * K + nDocSentiment[i][s]));
                double term2 = ((omega + nDocSentiment[i][s])/(omega * S + nTotalSentences[i]));
 
                for (int t = 0; t < totalPermutation; t++){
                    int[] indicators = convertToIntArray(t, numTokens);
                    
                    double term3 = 1;
                    // calculate the p(x|z,w,l,gamma)
                    for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++){
                        if (tokenIdx == 0){ // if starting
                            term3 = term3 * (((gamma + nPreSentimentPreTopicPreWordIndicator[s][k][V][indicators[tokenIdx]])
                                    /(gamma * I + nPreSentimentPreTopicPreWord[s][k][V])));
                        }else{
                            term3 = term3 * (((gamma + nPreSentimentPreTopicPreWordIndicator[s][k][sentence.getToken(tokenIdx-1)-1][indicators[tokenIdx]])
                                    /(gamma * I + nPreSentimentPreTopicPreWord[s][k][sentence.getToken(tokenIdx-1)])));
                        }
                    }
                    
                    double term4 = 1;
                    // calculate p(w|z,l,x,beta,delta)
                    for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++){
                        if (indicators[tokenIdx] == 0){
                            term4 = term4 * (((beta + nWordSentimentTopicWords[s][k][sentence.getToken(tokenIdx)])
                                    /(beta * V + nWordSentimentTopic[s][k])));
                        }else{
                            if (tokenIdx == 0){
                                term4 = term4 * (((delta + nSentimentTopicPrevWordWord[s][k][V][sentence.getToken(tokenIdx)])
                                        /(delta * V + nSentimentTopicPreWord[s][k][V])));
                            }else{
                                term4 = term4 * (((delta + nSentimentTopicPrevWordWord[s][k][sentence.getToken(tokenIdx-1)][sentence.getToken(tokenIdx)])
                                        /(delta * V + nSentimentTopicPreWord[s][k][sentence.getToken(tokenIdx-1)])));
                            }
                            
                        }
                    }
                    
                    p[s][k][t] = term1 * term2 * term3 * term4;
                    if (Double.isNaN(p[s][k][t] ) || Double.isInfinite(p[s][k][t])){
                        int aa =1;
                    }
                }
            }
        }
        
        
  
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSample(p);
        if (p.length == 0){
            int aaa =1;
        }
        int newTopic = latentVariable.getTopic();
        int newSentiment = latentVariable.getSentiment();
        int newIndicatorIdx = latentVariable.getIndicator();
        
        int[] newIndicators = convertToIntArray(newIndicatorIdx, numTokens);
        latentVariable.setIndicators(newIndicators);
        
        
        nDocSentimentTopic[i][newSentiment][newTopic]++;
        nDocSentiment[i][newSentiment]++;
        nTotalSentences[i]++;
        
        // iterate each word. 
        for (int k = 0; k < sentence.getTokens().size(); k++){
            int newIndicator = newIndicators[k];
            
            // update indicator related variables. 
            if (k == 0){ // beginning word
                nPreSentimentPreTopicPreWordIndicator[newSentiment][newTopic][V][newIndicator]++;
                nPreSentimentPreTopicPreWord[newSentiment][newTopic][V]++;
            }else{ 
                nPreSentimentPreTopicPreWordIndicator[newSentiment][newTopic][sentence.getToken(k-1)][newIndicator]++;
                nPreSentimentPreTopicPreWord[newSentiment][newTopic][sentence.getToken(k-1)]++;
            }
            
            // update word probability. 
            if (newIndicator == 0){ // use unigram
                nWordSentimentTopic[newSentiment][newTopic]++;
                nWordSentimentTopicWords[newSentiment][newTopic][sentence.getToken(k)]++;
            } else{ // use bigram. 
                if (k ==0){ // if begining word
                    nSentimentTopicPrevWordWord[newSentiment][newTopic][V][sentence.getToken(k)]++;
                    nSentimentTopicPreWord[newSentiment][newTopic][V]++;
                }else{
                    nSentimentTopicPrevWordWord[newSentiment][newTopic][sentence.getToken(k-1)][sentence.getToken(k)]++;
                    nSentimentTopicPreWord[newSentiment][newTopic][sentence.getToken(k-1)]++;
                }
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
    
    /*
     * convert val to binary of size n
     */
    private int[] convertToIntArray(int val, int n){
        String binaryString = Integer.toBinaryString(val);
        int[] result = new int[n];
        int j = result.length - 1;
        for (int i = binaryString.length()-1; i>=0; i--){
            if (binaryString.charAt(i) == '0'){
                result[j] = 0;
            }else{
                result[j] = 1;
            }
            j--;
        }
        
        return result;
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
