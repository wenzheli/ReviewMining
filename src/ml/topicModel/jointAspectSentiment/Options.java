package ml.topicModel.jointAspectSentiment;

// If not set by users, just use these default values. 
public class Options {
    public double alpha = 1;  // alpha is the prior for document-topic distributions
    public double beta = 0.01;   // beta is the prior for topic-word distributions. 
    public int niters = 100;    // number of Gibbs sampling iteration
    public int liter = 80;     // the iteration at which the model was saved  
    public int savestep = 10;  // saving period
    public int tWords = 10;    // # of top words to display
    public int K = 50;         // # of topics we learn from the model.
    public int S = 2;          // # of sentiments
}
