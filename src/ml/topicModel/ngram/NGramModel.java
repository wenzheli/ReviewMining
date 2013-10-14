package ml.topicModel.NGram;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ml.topicModel.NGSentimentSentence.SparseMatrix;
import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.data.LatentVariable;
import ml.topicModel.common.data.NGramDocument;
import ml.topicModel.common.data.WDocument;
import ml.topicModel.utils.DistributionUtils;


public class NGramModel {
 
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
    
    private int[][] z; // latent variable, topic assignments for each word. D * document.size()
    private int[][] x; // latent variable, indicator variable for denoting if the current term form bigram with the previous word
    
    private int [][] nTopicWords; // nTopicWords[i][j]: # of instances of word/term j assigned to topic i, size K*V
    private int [][] nDocTopic;   // nDocTopic[i][j]: # of words in document i that assigned to topic j, size D x K
    private int [] nWordTopic; // nWordTopic[j]: total number of words assigned to topic j, size K. 
    private int [] nWordsSum;     // nWordsSum[i]: total number of words in document i, size D
    private SparseMatrix [] nTopicPrevWordWord; //nTopicWordWord[i][j][k]:  total number of word/term k, assigned to topic i, on the condition that the previous
                                       // previous term is j. size: K * (V+1) * V
    private int [][] nTopicPreWord;  // nTopicPreWord[i][j]:  total number of word/term that assigned to topic i, where previous word is j. 
                                     // size: K *(V+1)
    private int [][][] nTopicPreWordIndicator;  // nTopicPreWordIndicator[i][j][k]: total number of indicator variable k, for the condition where
                                                 // the topic of previous word is i, and the previous term/word is j. 
    private int [][] nTopicPreWordIndicatorSum;  // size: K * (V+1) 
    
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
        nTopicWords = new int[K][V];
        nDocTopic = new int[D][K];
        nWordTopic = new int[K];
        nWordsSum = new int[D];
        nTopicPrevWordWord = new SparseMatrix[K];
        for (int i = 0; i < K; i++){
            nTopicPrevWordWord[i] = new SparseMatrix(V+1);
        }
        nTopicPreWord = new int[K][V+1];
        nTopicPreWordIndicator = new int[K+1][V+1][2];
        nTopicPreWordIndicatorSum = new int[K+1][V+1];
        
        // initialize latent variable - z and x
        z = new int[D][];
        x = new int[D][];
        for (int i = 0; i < D; i++){
            WDocument d = (WDocument) dataset.getDocument(i);
            int numTerms = d.getNumOfTokens();
            z[i] = new int[numTerms];
            x[i] = new int[numTerms];
            // for each word, randomly assign topic and indicator value. 
            for (int j = 0; j < numTerms; j++){
                int randTopic = (int)(Math.random() * K);
                int randIndicatorValue = 0;
                // if the term is starter word, it shouldn't be combine with the previous word. 
                if (j == 0)
                    randIndicatorValue = 0;
                else
                    randIndicatorValue = (int)(Math.random() * I);
                
                z[i][j] = randTopic;
                x[i][j] = randIndicatorValue;
                
                
                nWordsSum[i]++;
                nDocTopic[i][randTopic]++;
                
                // if starting word....
                if (j == 0){
                    nTopicPreWordIndicator[K][V][randIndicatorValue]++;
                    nTopicPreWordIndicatorSum[K][V]++;
                } else{
                    nTopicPreWordIndicator[z[i][j-1]][d.getToken(j-1)][randIndicatorValue]++;
                    nTopicPreWordIndicatorSum[z[i][j-1]][d.getToken(j-1)]++;
                }
                
                // use uni-gram 
                if (x[i][j] == 0){
                    nTopicWords[randTopic][d.getToken(j)]++;  
                    nWordTopic[randTopic]++;
                    
                } else{ // use bigram 
                    if (j == 0){ // if beginning term
                        nTopicPrevWordWord[randTopic].increment(V, d.getToken(j));
                        nTopicPreWord[randTopic][V]++;    
                    } else{ 
                        nTopicPrevWordWord[randTopic].increment(d.getToken(j-1),d.getToken(j));
                        nTopicPreWord[randTopic][d.getToken(j-1)]++;
                    }
                }
                  
            }
            
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runSampler(){
        for (int i = 0; i < D; i++){
            WDocument d = (WDocument) dataset.getDocument(i);
            for (int j = d.getNumOfTokens()-1; j >=0; j--){
                // random sample z[i][j] 
                LatentVariable latentVariable = sample(i,j);
                z[i][j] = latentVariable.getTopic();
                x[i][j] = latentVariable.getIndicatorVariable();
            }
        }
        
        int aaa =1;
    }
    
    private LatentVariable sample(int i,  int j){
        WDocument d = (WDocument) dataset.getDocument(i);
        int termCnt = d.getNumOfTokens();
        int oldTopic = z[i][j];
        int oldIndicatorValue = x[i][j];
        
        nWordsSum[i]--;      
        nDocTopic[i][oldTopic]--;
        
        // update nTopicPreWordIndicator and nTopicPreWordIndicatorSum
        if (j == 0){
            nTopicPreWordIndicator[K][V][oldIndicatorValue]--;
            nTopicPreWordIndicatorSum[K][V]--;
          
            // and not the last word
            if (j < termCnt - 1){
                nTopicPreWordIndicator[oldTopic][d.getToken(j)][x[i][j+1]]--;
                nTopicPreWordIndicatorSum[oldTopic][d.getToken(j)]--;
               
            }
        } else {
            if (nTopicPreWordIndicator[z[i][j-1]][d.getToken(j-1)][oldIndicatorValue] < 1){
                int aa =1;
            }
            nTopicPreWordIndicator[z[i][j-1]][d.getToken(j-1)][oldIndicatorValue]--;
            nTopicPreWordIndicatorSum[z[i][j-1]][d.getToken(j-1)]--;

            if (j < termCnt - 1){
                if (nTopicPreWordIndicator[oldTopic][d.getToken(j)][x[i][j+1]] < 0){
                    int aa =1;
                }
                nTopicPreWordIndicator[oldTopic][d.getToken(j)][x[i][j+1]]--;
                nTopicPreWordIndicatorSum[oldTopic][d.getToken(j)]--;
                
            }
        }
        
        // if unigram...
        if (oldIndicatorValue == 0){
            nTopicWords[oldTopic][d.getToken(j)]--;  
            nWordTopic[oldTopic]--;
        } else{ // if bigram
            if (j == 0){ // if beginning term
                nTopicPrevWordWord[oldTopic].decrement(V, d.getToken(j));
                nTopicPreWord[oldTopic][V]--; 
                
            } else{ 
                nTopicPrevWordWord[oldTopic].decrement(d.getToken(j-1), d.getToken(j));
                nTopicPreWord[oldTopic][d.getToken(j-1)]--;
            }
        }
        
      
        
        // compute p(z[i][j]|*)
        double[][] p = new double[K][I];
        for (int k = 0; k < K; k++){
            for (int s = 0; s < I; s++){
                // if uni-gram
                if (s == 0){
                    if (j == 0){
                        
                        p[k][s] = ((alpha + nDocTopic[i][k])/(K*alpha + nWordsSum[i])) 
                                * ((beta+nTopicWords[k][d.getToken(j)])/(V*beta+nWordTopic[k]))
                                * ((gamma + nTopicPreWordIndicator[K][V][s])/(I*gamma + nTopicPreWordIndicatorSum[K][V]));
                        
                    } else{
                        p[k][s] = ((alpha + nDocTopic[i][k])/(K*alpha + nWordsSum[i])) 
                                * ((beta+nTopicWords[k][d.getToken(j)])/(V*beta+nWordTopic[k]))
                                * ((gamma + nTopicPreWordIndicator[(z[i][j-1])][d.getToken(j-1)][s])/(I*gamma + nTopicPreWordIndicatorSum[z[i][j-1]][d.getToken(j-1)]));
                       
                    }
                    
                } else{
                    if (j == 0){
                        p[k][s] = ((alpha + nDocTopic[i][k])/(K*alpha + nWordsSum[i])) 
                                * ((delta+nTopicPrevWordWord[k].get(V, d.getToken(j)))/(V*delta+nTopicPreWord[k][V]))
                                * ((gamma + nTopicPreWordIndicator[K][V][s])/(I*gamma + nTopicPreWordIndicatorSum[K][V]));
                        
                    }else {
                        p[k][s] = ((alpha + nDocTopic[i][k])/(K*alpha + nWordsSum[i])) 
                                * ((delta+nTopicPrevWordWord[k].get(d.getToken(j-1), d.getToken(j)))/(V*delta+nTopicPreWord[k][d.getToken(j-1)]))
                                * ((gamma + nTopicPreWordIndicator[(z[i][j-1])][d.getToken(j-1)][s])/(I*gamma + nTopicPreWordIndicatorSum[z[i][j-1]][d.getToken(j-1)]));
                       
                    }
                }
                
            }
            
        }
        
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSampleNGram(p);
        int newTopic = latentVariable.getTopic();
        int newIndicatorValue = latentVariable.getIndicatorVariable();
        
        nWordsSum[i]++;      
        nDocTopic[i][newTopic]++;
        
        // update nTopicPreWordIndicator and nTopicPreWordIndicatorSum
        if (j == 0){
            nTopicPreWordIndicator[K][V][newIndicatorValue]++;
            nTopicPreWordIndicatorSum[K][V]++;
            // and not the last word
            if (j < termCnt - 1){
                nTopicPreWordIndicator[newTopic][d.getToken(j)][x[i][j+1]]++;
                nTopicPreWordIndicatorSum[newTopic][d.getToken(j)]++;
            }
        } else {
            nTopicPreWordIndicator[z[i][j-1]][d.getToken(j-1)][newIndicatorValue]++;
            nTopicPreWordIndicatorSum[z[i][j-1]][d.getToken(j-1)]++;
            if (j < termCnt - 1){
                nTopicPreWordIndicator[newTopic][d.getToken(j)][x[i][j+1]]++;
                nTopicPreWordIndicatorSum[newTopic][d.getToken(j)]++;
            }
        }
        
        // if unigram...
        if (newIndicatorValue == 0){
            nTopicWords[newTopic][d.getToken(j)]++;  
            nWordTopic[newTopic]++;
        } else{ // if bigram
            if (j == 0){ // if beginning term
                nTopicPrevWordWord[newTopic].increment(V, d.getToken(j));
                nTopicPreWord[newTopic][V]++;            
            } else{ 
                nTopicPrevWordWord[newTopic].increment(d.getToken(j-1), d.getToken(j));
                nTopicPreWord[newTopic][d.getToken(j-1)]++;
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
    
    // combine unigram and ngrams...
    public double[][] getWordDistribution(){
        List<NGramDocument> nGramDocuments = new ArrayList<NGramDocument>();
        for (int i = 0; i < D; i++){
            // for each document, we reconstruct vocabulary using learned x
            List<List<Integer>> terms = new ArrayList<List<Integer>>();
            List<Integer> topics = new ArrayList<Integer>();
            List<Integer> words;
            WDocument d = (WDocument) dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                if(x[i][j] == 0){
                    words = new ArrayList<Integer>();
                    words.add(d.getToken(j));
                    topics.add(z[i][j]);
                    terms.add(words);
                }else{
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
        
        // construct word distribution combining unigram and n-gram. 
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
        
        int size = nGramToIndexMap.keySet().size();
        
        double[][] topicNGram = new double[K][size];
        double[] topicTotal = new double[K];
        for (NGramDocument nGramDoc : nGramDocuments){
            List<List<Integer>> nGramTerms = nGramDoc.getTerms();
            List<Integer> nGramTopics = nGramDoc.getTopics();
            for (int i = 0; i < nGramTerms.size(); i++){
                int idx = nGramToIndexMap.get(nGramTerms.get(i));
                topicNGram[nGramTopics.get(i)][idx]++;
                topicTotal[nGramTopics.get(i)]++;
            }
        }
        
        double[][] topicNGramDist = new double[K][size];
        for (int k =0; k < K; k++){
            for (int i = 0; i < size; i++){
                topicNGramDist[k][i] = ((beta+topicNGram[k][i])/(beta *V + topicTotal[k]));
            }
        }
        
        return topicNGramDist;
        
    }
    
    public double[][] getTopWordsFromNGram(){
        List<NGramDocument> nGramDocuments = new ArrayList<NGramDocument>();
        
        for (int i = 0; i < D; i++){
            List<List<Integer>> terms = new ArrayList<List<Integer>>();
            List<Integer> topics = new ArrayList<Integer>();
            List<Integer> words;
            WDocument d = (WDocument) dataset.getDocument(i);
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
        int size = nGramToIndexMap.keySet().size();
        double[][] topicNGram = new double[K][size];
        double[] topicTotal = new double[K];
        double[][] topicNGramDist = new double[K][nGramToIndexMap.keySet().size()];
        for (NGramDocument nGramDoc : nGramDocuments){
            List<List<Integer>> nGramTerms = nGramDoc.getTerms();
            List<Integer> nGramTopics = nGramDoc.getTopics();
            for (int i = 0; i < nGramTerms.size(); i++){
                int idx = nGramToIndexMap.get(nGramTerms.get(i));
                topicNGram[nGramTopics.get(i)][idx]++;
                topicTotal[nGramTopics.get(i)]++;
            }
        }
        
        for (int k =0; k < K; k++){
            for (int i = 0; i < size; i++){
                topicNGramDist[k][i] = ((beta+topicNGram[k][i])/(beta *V + topicTotal[k]));
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
