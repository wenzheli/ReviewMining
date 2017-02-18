package ml.topicModel.LDASentence;

// If not set by users, just use these default values. 
public class Options {
    public double alpha = 1;  // alpha is the prior for document-topic distributions
    public double beta = 0.01;   // beta is the prior for topic-word distributions. 
    public double gamma = 1;    // gamma is the prior for topic-sentiment distribution
    public int niters = 2000;    // number of Gibbs sampling iteration  
    public int savestep = 200;  // saving period
    public int tWords = 30;    // # of top words to display
    public int K = 50;         // # of topics we learn from the model. 
    public int S = 2;           // # of sentiments. 
}
