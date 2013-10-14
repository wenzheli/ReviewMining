package ml.topicModel.apps;

import java.io.IOException;

public class main {
    public static void main(String[] args) throws IOException{
        // test LDA
        //ml.topicModel.LDA.Inference.executeYelpDataSet();
        //ml.topicModel.LDA.Inference.executeNIPSDataSet();
        
        // test LDASentence
        //ml.topicModel.LDASentence.Inference.executeNIPSDataSet();
        //ml.topicModel.LDASentence.Inference.executeYelpDataSet();
        
        // test LDASentiment
        //ml.topicModel.LDASentiment.Inference.executeNIPSDataSet();
        ml.topicModel.LDASentiment.Inference.executeYelpDataSet();
        
        // test LDASentimentSentence
        //ml.topicModel.LDASentimentSentence.Inference.executeNIPSDataSet();
        //ml.topicModel.LDASentimentSentence.Inference.executeYelpDataSet();
    }
}
