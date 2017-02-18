package ml.topicModel.NGramSentence;

public class LatentVariable {
    private int topic;
    private int indicator;
    private int sentiment;
    private int[] indicators;
    
    public LatentVariable(int topic, int indicator){
        this.topic = topic;
        this.indicator = indicator;
    }
    
    public int getSentiment(){
        return sentiment;
    }
    public int getTopic(){
        return topic;
    }
    
    public int getIndicator(){
        return indicator;
    }
    
    public void setIndicators(int[] indicators){
        this.indicators = indicators;
    }
    
    public int[] getIndicators(){
        return indicators;
    }
}
