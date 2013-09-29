package ml.topicModel.NGramSentiment;

public class LatentVariable {
    private int sentiment;
    private int topic;
    private int indicator;
    
    public LatentVariable(int sentiment,int topic, int indicator){
        this.sentiment = sentiment;
        this.topic = topic;
        this.indicator = indicator;
    }
    
    public int getTopic(){
        return topic;
    }
    
    public int getIndicator(){
        return indicator;
    }
    
    public int getSentiment(){
        return sentiment;
    }
}
