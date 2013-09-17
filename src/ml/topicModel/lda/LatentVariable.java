package ml.topicModel.lda;

public class LatentVariable {
    private int topic;
    private int indicator;
    
    public LatentVariable(int topic, int indicator){
        this.topic = topic;
        this.indicator = indicator;
    }
    
    public int getTopic(){
        return topic;
    }
    
    public int getIndicator(){
        return indicator;
    }
}
