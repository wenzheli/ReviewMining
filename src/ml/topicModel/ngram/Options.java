package ml.topicModel.NGram;

// If not set by users, just use these default values. 
public class Options {
    public double alpha = 1;  // alpha is the prior for document-topic distributions
    public double beta = 0.01;   // beta is the prior for topic-word distributions. 
    public double gamma = 0.1;   // gamma is the prior for indicator distribution 
    public double delta = 0.01;  // delta is the prior for bi-gram topic-word distribution
    public int niters = 10000;    // number of Gibbs sampling iteration
    public int liter = 80;     // the iteration at which the model was saved  
    public int savestep = 200;  // saving period
    public int tWords = 30;    // # of top words to display
    public int K = 50;         // # of topics we learn from the model. 
}
