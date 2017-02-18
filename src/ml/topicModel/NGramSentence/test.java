package ml.topicModel.NGramSentence;

public class test {
    public static int[] converToIntArray(int val, int n){
        String binaryString = Integer.toBinaryString(val);
        int[] result = new int[n];
        int j = result.length - 1;
        for (int i = binaryString.length()-1; i>=0; i--){
            if (binaryString.charAt(i) == '0'){
                result[j] = 0;
            }else{
                result[j] = 1;
            }
            j--;
        }
        
        return result;
    }
    
    public static void main(String[] args){
        int[] result = converToIntArray(11, 8);
        System.out.println(result);
    }
}
