package ml.topicModel.NGramSentence;

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
  
    private int K;  // # of topics
    private int D;  // # of documents in the data set
    private int V;  // vocabulary size
    private int I = 2;
    
    private double [][] theta; // document - topic distributions, size D x K
    private double [][] phi;   // topic-word distributions, size K x V
    private double [][] psi;   // topic-word indicator distribution, size K * (V+1) * 2
    private double [][] sigma; // topic-word-word distribution, K * (V+1) * V
    
    private int[][] z; // latent variable, topic assignments for each word. D * number of sentences
    private int[][][] x; // latent variable, indicator variable for denoting if the current term form bigram with the previous word
 
    // used for topic and sentiments.
    private byte [][] nDocTopic;   // D*K nDocSentimentTopic[d][k]: # of sentences in document d that assigned to topic k, and sentiment s
  
    private int [] nTotalSentences;     // nTotalSentences[i]: total number of sentences in document i, size D
    
    private int [] nWordTopic; //  K nWordTopic[k]: total number of words assigned to topic j, size K. 
    private int [][] nWordTopicWords; // K*V nSentimentTopicWords[s][k][w]: # of instances of word/term w assigned to topic k, and sentiment s, 
    
    
    // used for bigram
    private byte [][][] nTopicPrevWordWord; //[j][k]:  total number of word/term k, assigned to topic i, on the condition that the previous
                                       // previous term is j. size: K * (V+1) * V
    private short [][] nTopicPreWord;  // nTopicPreWord[i][j]:  total number of word/term that assigned to topic i, where previous word is j. 
                                     // size: K *(V+1)
    // used for indicator variable. 
    private short [][][] nPreTopicPreWordIndicator;  // nTopicPreWordIndicator[j][k]: total number of indicator variable k, for the condition where
                                                 // the topic of previous word is i, and the previous term/word is j. 
    private short [][] nPreTopicPreWord;  // size: K * (V+1) 
    
    private DataSet dataset;
    
    Map<List<Integer>, Integer> nGramToIndexMap;
    Map<Integer,List<Integer>> indexToNGramMap;
    
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        this.alpha = options.alpha;
        this.beta = options.beta;
        this.gamma = options.gamma;
        this.delta = options.delta;
   
        
        this.K = options.K;
        this.D = dataset.getDocumentCount();
        this.V = dataset.getVocabulary().getVocabularySize();
        
        this.dataset = dataset;
        
        // these parameters are sufficient statistics of latent variable Z. We only sample z instead
        theta = new double[D][K];
        phi = new double[K][V];
        
        // initialize temporary variables
        nDocTopic = new byte[D][K];
        nTotalSentences = new int[D];
        nWordTopic = new int[K];
        nWordTopicWords = new int[K][V];
        
        nTopicPrevWordWord = new byte[K][V+1][V];
        nTopicPreWord = new short[K][V+1];
        nPreTopicPreWordIndicator = new short[K][V+1][I];
        nPreTopicPreWord = new short[K][V+1];
        
        // initialize latent variable - z and x
        // i: document  j: sentence   k: words
        z = new int[D][];
        x = new int[D][][];
        
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            // assign topic and sentiment to each sentence. 
            z[i] = new int[numSentences];
            x[i] = new int[numSentences][];
            
            // iterate each sentence. 
            for (int j = 0; j < numSentences; j++){
                // assign topic and sentiment for each sentence. 
                int randTopic = (int)(Math.random() * K);  
                z[i][j] = randTopic;
      
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
                        nPreTopicPreWordIndicator[randTopic][V][randIndicator]++;
                        nPreTopicPreWord[randTopic][V]++;
                    }else{ 
                        nPreTopicPreWordIndicator[randTopic][sentence.getToken(k-1)][randIndicator]++;
                        nPreTopicPreWord[randTopic][sentence.getToken(k-1)]++;
                    }
                    
                    // update word probability. 
                    if (randIndicator == 0){ // use unigram
                        nWordTopic[randTopic]++;
                        nWordTopicWords[randTopic][sentence.getToken(k)]++;
                    } else{ // use bigram. 
                        if (k ==0){ // if begining word
                            nTopicPrevWordWord[randTopic][V][sentence.getToken(k)]++;
                            nTopicPreWord[randTopic][V]++;
                        }else{
                            nTopicPrevWordWord[randTopic][sentence.getToken(k-1)][sentence.getToken(k)]++;
                            nTopicPreWord[randTopic][sentence.getToken(k-1)]++;
                        }
                    }
                    
                }
                
                nDocTopic[i][randTopic]++;
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
                x[i][j] = latentVariable.getIndicators();
            }
            
            //System.out.println(i);
        }
        
        
    }
    
    // sampling z,l,x for sentence j, in document i. 
    private LatentVariable sample(int i,  int j){
        Document d = dataset.getDocument(i);
        Sentence sentence = d.getSentence(j);
      
        int oldTopic = z[i][j];
        int[] oldIndicators = x[i][j];
        
        nDocTopic[i][oldTopic]--;
        nTotalSentences[i]--;
        
        // iterate each word. 
        for (int k = 0; k < sentence.getTokens().size(); k++){
            int oldIndicator = oldIndicators[k];
            
            // update indicator related variables. 
            if (k == 0){ // beginning word
                nPreTopicPreWordIndicator[oldTopic][V][oldIndicator]--;
                nPreTopicPreWord[oldTopic][V]--;
            }else{ 
                nPreTopicPreWordIndicator[oldTopic][sentence.getToken(k-1)][oldIndicator]--;
                nPreTopicPreWord[oldTopic][sentence.getToken(k-1)]--;
            }
            
            // update word probability. 
            if (oldIndicator == 0){ // use unigram
                nWordTopic[oldTopic]--;
                nWordTopicWords[oldTopic][sentence.getToken(k)]--;
            } else{ // use bigram. 
                if (k ==0){ // if begining word
                    nTopicPrevWordWord[oldTopic][V][sentence.getToken(k)]--;
                    nTopicPreWord[oldTopic][V]--;
                }else{
                    nTopicPrevWordWord[oldTopic][sentence.getToken(k-1)][sentence.getToken(k)]--;
                    nTopicPreWord[oldTopic][sentence.getToken(k-1)]--;
                }
            }
        }
        
        
        
        /*
         *  compute p(z[i][j]|*)
         */
        int numTokens = sentence.getTokens().size();
        // generate random permutation
        int totalPermutation = (int) Math.pow(2, numTokens-1);
        
        double[][] p = new double[K][totalPermutation];
        
      
            for (int k = 0; k < K; k++){
                double term1 = ((alpha + nDocTopic[i][k])/(alpha * K + nTotalSentences[i]));

                for (int t = 0; t < totalPermutation; t++){
                    int[] indicators = convertToIntArray(t, numTokens);
                    
                    double term3 = 1;
                    // calculate the p(x|z,w,l,gamma)
                    for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++){
                        if (tokenIdx == 0){ // if starting
                            term3 = term3 * (((gamma + nPreTopicPreWordIndicator[k][V][indicators[tokenIdx]])
                                    /(gamma * I + nPreTopicPreWord[k][V])));
                        }else{
                            term3 = term3 * (((gamma + nPreTopicPreWordIndicator[k][sentence.getToken(tokenIdx-1)-1][indicators[tokenIdx]])
                                    /(gamma * I + nPreTopicPreWord[k][sentence.getToken(tokenIdx-1)])));
                        }
                    }
                    
                    double term4 = 1;
                    // calculate p(w|z,l,x,beta,delta)
                    for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++){
                        if (indicators[tokenIdx] == 0){
                            term4 = term4 * (((beta + nWordTopicWords[k][sentence.getToken(tokenIdx)])
                                    /(beta * V + nWordTopic[k])));
                        }else{
                            if (tokenIdx == 0){
                                term4 = term4 * (((delta + nTopicPrevWordWord[k][V][sentence.getToken(tokenIdx)])
                                        /(delta * V + nTopicPreWord[k][V])));
                            }else{
                                term4 = term4 * (((delta + nTopicPrevWordWord[k][sentence.getToken(tokenIdx-1)][sentence.getToken(tokenIdx)])
                                        /(delta * V + nTopicPreWord[k][sentence.getToken(tokenIdx-1)])));
                            }
                            
                        }
                    }
                    
                    p[k][t] = term1 * term3 * term4;
                }
            }
      
        
        
  
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSample(p);
        
        int newTopic = latentVariable.getTopic();
        int newIndicatorIdx = latentVariable.getIndicator();
        
        int[] newIndicators = convertToIntArray(newIndicatorIdx, numTokens);
        latentVariable.setIndicators(newIndicators);
        
       
        nDocTopic[i][newTopic]++;
        nTotalSentences[i]++;
        
        // iterate each word. 
        for (int k = 0; k < sentence.getTokens().size(); k++){
            int newIndicator = newIndicators[k];
            
            // update indicator related variables. 
            if (k == 0){ // beginning word
                nPreTopicPreWordIndicator[newTopic][V][newIndicator]++;
                nPreTopicPreWord[newTopic][V]++;
            }else{ 
                nPreTopicPreWordIndicator[newTopic][sentence.getToken(k-1)][newIndicator]++;
                nPreTopicPreWord[newTopic][sentence.getToken(k-1)]++;
            }
            
            // update word probability. 
            if (newIndicator == 0){ // use unigram
                nWordTopic[newTopic]++;
                nWordTopicWords[newTopic][sentence.getToken(k)]++;
            } else{ // use bigram. 
                if (k ==0){ // if begining word
                    nTopicPrevWordWord[newTopic][V][sentence.getToken(k)]++;
                    nTopicPreWord[newTopic][V]++;
                }else{
                    nTopicPrevWordWord[newTopic][sentence.getToken(k-1)][sentence.getToken(k)]++;
                    nTopicPreWord[newTopic][sentence.getToken(k-1)]++;
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
