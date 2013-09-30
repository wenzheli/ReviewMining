package ml.topicModel.common.data;

public class LatentVariable {
    private int topic;
    private int sentiment;
    private int indicatorVariable;
    private int[] indicators;
     
    public int getTopic(){
        return topic;
    }
    
    public int getSentiment(){
        return sentiment;
    }
    
    public int getIndicatorVariable(){
        return indicatorVariable;
    }
    
    public void setTopic(int topic){
        this.topic = topic;
    }
    
    public void setSentiment(int sentiment){
        this.sentiment = sentiment;
    }
    
    public void setIndicatorVariable(int indicatorVariable){
        this.indicatorVariable = indicatorVariable;
    }
    
    public int[] getIndicators(){
        return indicators;
    }
    
    public void setIndicators(int[] indicators){
        this.indicators = indicators;
    }
}
