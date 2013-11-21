package ml.topicModel.DCMLDASentiment;

public class Options {
    public double alpha = 1;     // alpha is the prior for document-topic distributions
    public double beta = 0.01;   // beta is the prior for topic-word distributions. 
    public double gamma = 1;
    public int burnin = 500;     // number of iterations before we start sampling
    public int sampleWait = 20;  // number of iterations for per sampling period
    public int samples = 50;     // number of sampling period
    public int tWords = 20;      // number of top words to display
    public int K = 50;           // number topics we learn from the model.
    public int S = 2;
    public String inputFile = "data/nipstxt"; // by default, use this for testing. 
}
