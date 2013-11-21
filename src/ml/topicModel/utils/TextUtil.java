package ml.topic.utils;

/**
 * Text-related utility functions. 
 * @author wenzhe  
 *
 */
public class TextUtil {
    /**
     * Remove non-letter or digit characters from the beginning and end of the token. 
     * i.e "~good!" -> "good"
     * @param token  input token
     * @return       processed token.  
     */
    public static String removeSpecialCharacters(String token){
        int start = 0; 
        int end = token.length()-1;
        for (int i = 0; i < token.length(); i++){
            char ch = token.charAt(i);
            if (!Character.isLetterOrDigit(ch))
                start++;
            else
                break;
        }
        
        for (int i = token.length()-1; i >=0; i--){
            char ch = token.charAt(i);
            if (!Character.isLetterOrDigit(ch))
                end--;
            else
                break;
        }
        
        if (start < end){
            return token.substring(start, end+1);
        } else
            return "";
    }
    
    /**
     * Check the token whether it is valid or not.
     * i.e "NLP" -> true,  "123" -> false 
     * @param token         the input token
     * @return              true if valid token, false otherwise. 
     */
    public static boolean isValid(String token){
        // ignore any token with the length less than 3. 
        if (token.length() < 3){
            return false;
        }
        // if token contains digit, then mark it as invalid. 
        int count = 0;
        for (int i = 0; i < token.length(); i++){
            char ch = token.charAt(i);
            if (Character.isDigit(ch)){
                return false;
            }
            
            if (Character.isLetter(ch)){
                count++;
            }
        }
        // if token contains less than 3 letters, mark it as invalid. 
        if (count < 3){
            return false;
        }  
        
        return false;
    }
}
