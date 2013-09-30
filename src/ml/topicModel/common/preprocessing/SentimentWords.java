package ml.topicModel.common.preprocessing;

import java.util.HashSet;
import java.util.Set;

public class SentimentWords {
    private Set<String> positive;
    private Set<String> negative;
    
    public SentimentWords(){
        positive = new HashSet<String>();
        negative = new HashSet<String>();
        
        positive.add("good");
        positive.add("nice");
        positive.add("excellent");
        positive.add("positive");
        positive.add("fortunate");
        positive.add("correct");
        positive.add("superior");
        positive.add("amazing");
        positive.add("attractive");
        positive.add("awesome");
        positive.add("best");
        positive.add("comfortable");
        positive.add("enjoy");
        positive.add("fantastic");
        positive.add("favorite");
        positive.add("fun");
        positive.add("glad");
        positive.add("great");
        positive.add("happy");
        positive.add("impressive");
        positive.add("love");
        positive.add("perfect");
        positive.add("recommend");
        positive.add("satisfied");
        positive.add("thank");
        positive.add("worth");
       
        negative.add("bad");
        negative.add("nasty");
        negative.add("negative");
        negative.add("unfortunate");
        negative.add("wrong");
        negative.add("inferior");
        negative.add("annoying");
        negative.add("complain");
        negative.add("disappointed");
        negative.add("hate");
        negative.add("junk");
        negative.add("mess");
        negative.add("not good");
        negative.add("not like");
        negative.add("not recommend");
        negative.add("not worth");
        negative.add("problem");
        negative.add("regret");
        negative.add("sorry");
        negative.add("terrible");
        negative.add("trouble");
        negative.add("unacceptable");
        negative.add("upset");
        negative.add("waste");
        negative.add("worst");
        negative.add("worthless");
    }
    
    public Set<String> getPositiveWords(){
        return positive;
    }
    
    public Set<String> getNegativeWords(){
        return negative;
    }
}
