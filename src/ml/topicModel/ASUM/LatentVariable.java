package ml.topicModel.ASUM;

public class LatentVariable {
    private int topic;
    private int sentiment;
    
    public LatentVariable(int topic, int sentiment){
        this.topic = topic;
        this.sentiment = sentiment;
    }
    
    public int getTopic(){
        return topic;
    }
    
    public int getSentiment(){
        return sentiment;
    }
}
