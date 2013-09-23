ReviewMining
============

Topic models for review mining


It includes several features. 

- LDA for aspect mining
- N-gram topic model for aspect mining
- Sentence LDA for aspect mining
- Sentence N-gram topic model for aspect mining
- Joint sentiment-LDA model for senti-aspect mining
- Joint sentiment-NGram model for senti-aspect mining
- Joint sentence sentiment-LDA model for aspect mining
- Joint sentence sentiment-N-gram model for aspect mining. 


1. LDA for aspect mining
This is classifical approach for topic modeling. The process is:  
for each document(customer review), we draw topic distribution, and draw each word from this topic distribution. 

Result will come soon


2. N-gram topic model for aspect mining
LDA is unigram model. It generates all the unigrams from each topic. However, it's also necessary to get n-grams from each topic. 
For this purpose, we maintain another latent variable X (beside Z), which is indicator variable that denotes whether we
need to concatenate current word with previous word.  The model can generate all n-gram words (unigram, bigram, tri-gram..etc)

Result will come soon

3. Sentence-LDA for aspect mining
This is slight modification of LDA model. For customer reviews, LDA shown to tend to generate global topics. For example, for customer review
for resturant, LDA tends to generate topic like "resturant" which is not aspect. Sentence-LDA will put constraint on the 
LDA model that each sentence comes from one aspect, and we draw all the words contained in that sentence from the same aspect ONLY.

Result will come soon

4. Sentence-Ngram for aspect mining
Similar to sentence-LDA, it is an extension of N-gram topic model 

Result will come soon










