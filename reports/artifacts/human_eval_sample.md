# Human Evaluation Sample (24 items)

- Baseline run: `hybrid_rerank.baseline.20260413T002827Z.json`
- Candidate run: `hybrid_rerank.citation_forcing.20260413T014339Z.json`

## 57f23dfc264feb62f45d9a9e24c60bd73d7fe563 [large_disagreement]

- Question: what is the size of the augmented dataset?
- Gold answers: `unanswerable || 609`
- Gold evidence: `FLOAT SELECTED: Table 1: Speech datasets used. Note that HAPD, HAFP and FP only have samples from healthy subjects. Detailed description in App. 2. || All datasets shown in Tab. SECREF2 were transcribed manually by trained transcriptionists, employing the same list of annotations and protocols, with the same set of features extracted from the transcripts (see Sec. SECREF3 ). HAPD and HAFP are jointly referred to as HA. || Binary classification of each speech transcript as AD or HC is performed. We do 5-fold cross-validation, stratified by subject so that each subject's samples do not occur in both training and testing sets in each fold. The minority class is oversampled in the training set using SMOTE BIBREF14 to deal with the class imbalance. We consider a Random Forest (100 trees), Naïve Bayes (with equal priors), SVM (with RBF kernel), and a 2-layer neural network (10 units, Adam optimizer, 500 epochs) BIBREF15 . Additionally, we augment the DB data with healthy samples from FP with varied ages. || We augment DB with healthy samples from FP with varying ages (Tab. SECREF11 ), considering 50 samples for each 15 year duration starting from age 30. Adding the same number of samples from bins of age greater than 60 leads to greater increase in performance. This could be because the average age of participants in the datasets (DB, HA etc.) we use are greater than 60. Note that despite such a trend, addition of healthy data produces fair classifiers with respect to samples with age INLINEFORM0 60 and those with age INLINEFORM1 60 (balanced F1 scores of 75.6% and 76.1% respectively; further details in App. SECREF43 .) || FLOAT SELECTED: Table 3: Augmenting DB with healthy data of varied ages. Scores averaged across 4 classifiers.`
- Baseline answer: `unanswerable`
- Candidate answer: `8.5 hours of total audio`
- Baseline metrics: F1=1.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[2] DementiaBank (DB): The DementiaBank dataset is the largest available public dataset of speech for assessing cognitive impairments. It consists of narrative picture descriptions from participants aged between 45 to 90 BIBREF24 . In each sample, a participant describes the picture that they are shown. Out of the 210 participants in the study, 117 were diagnosed with AD ( INLINEFORM0 samples of speech) and 93 were healthy (HC; INLINEFORM1 samples) with many subjects repeating the task with an in...
```
- Candidate NLI claim summary:

```text
[2] supported=no entail=0.001 :: 8.5 hours of total audio [2]
```

## 8e12b5c459fa963b3e549deadb864c244879fe82 [large_disagreement]

- Question: How many layers does their model have?
- Gold answers: `unanswerable`
- Baseline answer: `unanswerable`
- Candidate answer: `The model has 2 layers`
- Baseline metrics: F1=1.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[4] Data. We build a dataset of Twitter accounts based on two lists annotated in previous works. For the non-factual accounts, we rely on a list of 180 Twitter accounts from BIBREF1. This list was created based on public resources where suspicious Twitter accounts were annotated with the main fake news types (clickbait, propaganda, satire, and hoax). We discard the satire labeled accounts since their intention is not to mislead or deceive. On the other hand, for the factual accounts, we use a lis...
```
- Candidate NLI claim summary:

```text
[4] supported=no entail=0.003 :: The model has 2 layers [4]
```

## 8bb0011ad1d63996d5650770f3be18abdd9f7fc6 [large_disagreement]

- Question: Do they report results only on English datasets?
- Gold answers: `unanswerable || yes`
- Gold evidence: `A promising strategy to bridge the gap mentioned above is to integrate the neural networks of MRC models with the general knowledge of human beings. To this end, it is necessary to solve two problems: extracting general knowledge from passage-question pairs and utilizing the extracted general knowledge in the prediction of answer spans. The first problem can be solved with knowledge bases, which store general knowledge in structured forms. A broad variety of knowledge bases are available, such as WordNet BIBREF7 storing semantic knowledge, ConceptNet BIBREF8 storing commonsense knowledge, and Freebase BIBREF9 storing factoid knowledge. In this paper, we limit the scope of general knowledge to inter-word semantic connections, and thus use WordNet as our knowledge base. The existing way to solve the second problem is to encode general knowledge in vector space so that the encoding results can be used to enhance the lexical or contextual representations of words BIBREF10 , BIBREF11 . However, this is an implicit way to utilize general knowledge, since in this way we can neither understand nor control the functioning of general knowledge. In this paper, we discard the existing implicit way and instead explore an explicit (i.e. understandable and controllable) way to utilize general knowledge. || WordNet is a lexical database of English, where words are organized into synsets according to their senses. A synset is a set of words expressing the same sense so that a word having multiple senses belongs to multiple synsets, with each synset corresponding to a sense. Synsets are further related to each other through semantic relations. According to the WordNet interface provided by NLTK BIBREF12 , there are totally sixteen types of semantic relations (e.g. hypernyms, hyponyms, holonyms, meronyms, attributes, etc.). Based on synset and semantic relation, we define a new concept: semantic relation chain. A semantic relation chain is a concatenated sequence of semantic relations, which links a synset to another synset. For example, the synset “keratin.n.01” is related to the synset “feather.n.01” through the semantic relation “substance holonym”, the synset “feather.n.01” is related to the synset “bird.n.01” through the semantic relation “part holonym”, and the synset “bird.n.01” is related to the synset “parrot.n.01” through the semantic relation “hyponym”, thus “substance holonym INLINEFORM0 part holonym INLINEFORM1 hyponym” is a semantic relation chain, which links the synset “keratin.n.01” to the synset “parrot.n.01”. We name each semantic relation in a semantic relation chain as a hop, therefore the above semantic relation chain is a 3-hop chain. By the way, each single semantic relation is equivalent to a 1-hop chain.`
- Baseline answer: `no`
- Candidate answer: `unanswerable`
- Baseline metrics: F1=0.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=1.0000, support=1.0000, cite=0.0000

## 384bf1f55c34b36cb03f916f50bbefade6c86a75 [large_disagreement]

- Question: Does their model suffer exhibit performance drops when incorporating word importance?
- Gold answers: `no`
- Baseline answer: `no`
- Candidate answer: `yes`
- Baseline metrics: F1=1.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[2] We evaluate the effectiveness of estimating word importance by the translation performance decrease. More specifically, unlike the usual way, we measure the decrease of translation performance when perturbing a set of important words that are of top-most word importance in a sentence. The more translation performance degrades, the more important the word is. We use the standard BLEU score as the evaluation metric for translation performance. To make the conclusion more convincing, we conduct...

[3] Our main contributions are: Our study demonstrates the necessity and effectiveness of exploiting the intermediate gradients for estimating word importance. We find that word importance is useful for understanding NMT by identifying under-translated words. We provide empirical support for the design principle of NMT architectures: essential inductive bias (e.g., language characteristics) should be considered for model design.
```
- Candidate NLI claim summary:

```text
[2,3] supported=no entail=0.001 :: yes [2, 3]
```

## 8d074aabf4f51c8455618c5bf7689d3f62c4da1d [large_disagreement]

- Question: What are the limitations of existing Vietnamese word segmentation systems?
- Gold answers: `ambiguous words unknown words || lacks of complete review approaches, datasets and toolkits`
- Gold evidence: `Maximum matching (MM) is one of the most popular fundamental and structural segmentation algorithms for word segmentation BIBREF19 . This method is also considered as the Longest Matching (LM) in several research BIBREF9 , BIBREF3 . It is used for identifying word boundary in languages like Chinese, Vietnamese and Thai. This method is a greedy algorithm, which simply chooses longest words based on the dictionary. Segmentation may start from either end of the line without any difference in segmentation results. If the dictionary is sufficient BIBREF19 , the expected segmentation accuracy is over 90%, so it is a major advantage of maximum matching . However, it does not solve the problem of ambiguous words and unknown words that do not exist in the dictionary. || According to our observation, we found that is lacks of complete review approaches, datasets and toolkits which we recently used in Vietnamese word segmentation. A all sided review of word segmentation will help next studies on Vietnamese natural language processing tasks have an up-to-date guideline and choose the most suitable solution for the task. The remaining part of the paper is organized as follows. Section II discusses building corpus in Vietnamese, containing linguistic issues and the building progress. Section III briefly mentions methods to model sentences and text in machine learning systems. Next, learning models and approaches for labeling and segmenting sequence data will be presented in Section IV. Section V mainly addresses two existing toolkits, vnTokenizer and JVnSegmenter, for Vietnamese word segmentation. Several experiments based on mentioned approaches and toolkits are described in Section VI. Finally, conclusions and future works are given in Section VII.`
- Baseline answer: `unanswerable`
- Candidate answer: `Existing Vietnamese word segmentation systems lack complete review approaches, datasets, and toolkits.`
- Baseline metrics: F1=0.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.6000, support=1.0000, cite=0.5000
- Candidate cited context:

```text
[4] According to our observation, we found that is lacks of complete review approaches, datasets and toolkits which we recently used in Vietnamese word segmentation. A all sided review of word segmentation will help next studies on Vietnamese natural language processing tasks have an up-to-date guideline and choose the most suitable solution for the task. The remaining part of the paper is organized as follows. Section II discusses building corpus in Vietnamese, containing linguistic issues and t...

[5] There are several studies for Vietnamese Word Segmentation during last decade. For instance, Dinh et al. started the word segmentation task for Vietnamese with Neural Network and Weighted Finite State Transducer (WFST) BIBREF9 . Nguyen et al. continued with machine learning approaches, Conditional Random Fields and Support Vector Machine BIBREF7 . Most of statistical approaches are based on the architecture as shown in Figure 2. According to the architecture, recent studies and systems focus...
```
- Candidate NLI claim summary:

```text
[4,5] supported=no entail=0.124 :: Existing Vietnamese word segmentation systems lack complete review approaches, datasets, and toolkits [4, 5].
```

## a6d3e57de796172c236e33a6ceb4cca793dc2315 [large_disagreement]

- Question: What are the baselines?
- Gold answers: `unanswerable || Answer with content missing: (Experimental Setup missing subsections) To be selected: We compared REFRESH against a baseline which simply selects the first m leading sentences from each document (LEAD) and two neural models similar to ours (see left block in Figure 1), both trained with cross-entropy loss. Answer: LEAD`
- Gold evidence: `Experimental Setup || In this section we present our experimental setup for assessing the performance of our model which we call Refresh as a shorthand for REinFoRcement Learning-based Extractive Summarization. We describe our datasets, discuss implementation details, our evaluation protocol, and the systems used for comparison.`
- Baseline answer: `unanswerable`
- Candidate answer: `Traditional summarization methods using features to rank sentences`
- Baseline metrics: F1=1.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0727, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[1] Traditional summarization methods manually define features to rank sentences for their salience in order to identify the most important sentences in a document or set of documents BIBREF42 , BIBREF43 , BIBREF44 , BIBREF45 , BIBREF46 , BIBREF47 . A vast majority of these methods learn to score each sentence independently BIBREF48 , BIBREF49 , BIBREF50 , BIBREF51 , BIBREF52 , BIBREF53 , BIBREF54 and a summary is generated by selecting top-scored sentences in a way that is not incorporated into...
```
- Candidate NLI claim summary:

```text
[1] supported=no entail=0.013 :: Traditional summarization methods using features to rank sentences [1]
```

## 1b72aa2ec3ce02131e60626639f0cf2056ec23ca [citation_better]

- Question: How long is the dataset for each step of hierarchy?
- Gold answers: `Level A: 14100 Tweets Level B: 4640 Tweets Level C: 4089 Tweets`
- Gold evidence: `The data included in OLID has been collected from Twitter. We retrieved the data using the Twitter API by searching for keywords and constructions that are often included in offensive messages, such as `she is' or `to:BreitBartNews'. We carried out a first round of trial annotation of 300 instances with six experts. The goal of the trial annotation was to 1) evaluate the proposed tagset; 2) evaluate the data retrieval method; and 3) create a gold standard with instances that could be used as test questions in the training and test setting annotation which was carried out using crowdsourcing. The breakdown of keywords and their offensive content in the trial data of 300 tweets is shown in Table TABREF14 . We included a left (@NewYorker) and far-right (@BreitBartNews) news accounts because there tends to be political offense in the comments. One of the best offensive keywords was tweets that were flagged as not being safe by the Twitter `safe' filter (the `-' indicates `not safe'). The vast majority of content on Twitter is not offensive so we tried different strategies to keep a reasonable number of tweets in the offensive class amounting to around 30% of the dataset including excluding some keywords that were not high in offensive content such as `they are` and `to:NewYorker`. Although `he is' is lower in offensive content we kept it as a keyword to avoid gender bias. In addition to the keywords in the trial set, we searched for more political keywords which tend to be higher in offensive content, and sampled our dataset such that 50% of the the tweets come from political keywords and 50% come from non-political keywords. In addition to the keywords `gun control', and `to:BreitbartNews', political keywords used to collect these tweets are `MAGA', `antifa', `conservative' and `liberal'. We computed Fliess' INLINEFORM0 on the trial set for the five annotators on 21 of the tweets. INLINEFORM1 is .83 for Layer A (OFF vs NOT) indicating high agreement. As to normalization and anonymization, no user metadata or Twitter IDs have been stored, and URLs and Twitter mentions have been substituted to placeholders. We follow prior work in related areas (burnap2015cyber,davidson2017automated) and annotate our data using crowdsourcing using the platform Figure Eight. We ensure data quality by: 1) we only received annotations from individuals who were experienced in the platform; and 2) we used test questions to discard annotations of individuals who did not reach a certain threshold. Each instance in the dataset was annotated by multiple annotators and inter-annotator agreement has been calculated. We first acquired two annotations for each instance. In case of 100% agreement, we considered these as acceptable annotations, and in case of disagreement, we requested more annotations until the agreement was above 66%. After the crowdsourcing annotation, we used expert adjudication to guarantee the quality of the annotation. The breakdown of the data into training and testing for the labels from each level is shown in Table TABREF15 . || FLOAT SELECTED: Table 3: Distribution of label combinations in OLID.`
- Baseline answer: `unanswerable`
- Candidate answer: `The dataset contains 14,100 tweets`
- Baseline metrics: F1=0.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.2667, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[3] This paper presents OLID, a new dataset with annotation of type and target of offensive language. OLID is the official dataset of the shared task SemEval 2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval) BIBREF16 . In OffensEval, each annotation level in OLID is an independent sub-task. The dataset contains 14,100 tweets and is released freely to the research community. To the best of our knowledge, this is the first dataset to contain annotation of typ...
```
- Candidate NLI claim summary:

```text
[3] supported=no entail=0.002 :: The dataset contains 14,100 tweets [3]
```

## cf171fad0bea5ab985c53d11e48e7883c23cdc44 [citation_better]

- Question: What details are given about the Twitter dataset?
- Gold answers: `Those tweets are mostly much noisier and shorter compared to the reviews in the movie corpus. In total, there are 1,716 tweets. 973 of them are negative and 743 of them are positive. || one of the Twitter datasets is about Turkish mobile network operators, there are positive, neutral and negative labels and provide the total amount plus the distribution of labels`
- Gold evidence: `The second Turkish dataset is the Twitter corpus which is formed of tweets about Turkish mobile network operators. Those tweets are mostly much noisier and shorter compared to the reviews in the movie corpus. In total, there are 1,716 tweets. 973 of them are negative and 743 of them are positive. These tweets are manually annotated by two humans, where the labels are either positive or negative. We measured the Cohen's Kappa inter-annotator agreement score to be 0.82. If there was a disagreement on the polarity of a tweet, we removed it. || We also utilised two other datasets in English to test the cross-linguality of our approaches. One of them is a movie corpus collected from the web. There are 5,331 positive reviews and 5,331 negative reviews in this corpus. The other is a Twitter dataset, which has nearly 1.6 million tweets annotated through a distant supervised method BIBREF14. These tweets have positive, neutral, and negative labels. We have selected 7,020 positive tweets and 7,020 negative tweets randomly to generate a balanced dataset.`
- Baseline answer: `unanswerable`
- Candidate answer: `The English Twitter dataset has nearly 1.6 million tweets annotated, with accuracies decreasing when utilizing polarity labels.`
- Baseline metrics: F1=0.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0976, support=1.0000, cite=0.5000
- Candidate cited context:

```text
[2] We utilised two datasets for both Turkish and English to evaluate our methods. For Turkish, as the first dataset, we utilised the movie reviews which are collected from a popular website. The number of reviews in this movie corpus is 20,244 and the average number of words in reviews is 39. Each of these reviews has a star-rating score which is indicative of sentiment. These polarity scores are between the values 0.5 and 5, at intervals of 0.5. We consider a review to be negative it the score...

[4] We have demonstrated that using word vectors that capture only semantic and syntactic characteristics may be improved by taking into account their sentimental aspects as well. Our approaches are cross-lingual and cross-domain. They can be applied to other domains and other languages than Turkish and English with minor changes. Our study is one of the few ones that perform sentiment analysis in Turkish and leverages sentimental characteristics of words in generating word vectors and outperform...
```
- Candidate NLI claim summary:

```text
[2,4] supported=no entail=0.002 :: The English Twitter dataset has nearly 1.6 million tweets annotated, with accuracies decreasing when utilizing polarity labels [2, 4].
```

## 357eb9f0c07fa45e482d998a8268bd737beb827f [citation_better]

- Question: What baseline models are used?
- Gold answers: `the Poly-encoder from BIBREF7 humeau2019real Feed Yourself BIBREF13 is an open-domain dialogue agent with a self-feeding model Kvmemnn BIBREF14 is a key-value memory network with a knowledge base that uses a key-value retrieval mechanism to train over multiple domains simultaneously We compare against four dialogue system baselines: Kvmemnn, Feed Yourself, Poly-encoder, and a BERT bi-ranker baseline trained on the Persona-Chat dataset using the same training hyperparameters (including learning rate scheduler and length capping settings) described in Section SECREF20. a BERT bi-ranker || Kvmemnn Feed Yourself Poly-encoder BERT bi-ranker`
- Gold evidence: `Open-domain chatbots are more generic dialogue systems. An example is the Poly-encoder from BIBREF7 humeau2019real. It outperforms the Bi-encoder BIBREF8, BIBREF9 and matches the performance of the Cross-encoder BIBREF10, BIBREF11 while maintaining reasonable computation time. It performs strongly on downstream language understanding tasks involving pairwise comparisons, and demonstrates state-of-the-art results on the ConvAI2 challenge BIBREF12. Feed Yourself BIBREF13 is an open-domain dialogue agent with a self-feeding model. When the conversation goes well, the dialogue becomes part of the training data, and when the conversation does not, the agent asks for feedback. Lastly, Kvmemnn BIBREF14 is a key-value memory network with a knowledge base that uses a key-value retrieval mechanism to train over multiple domains simultaneously. We use all three of these models as baselines for comparison. While these can handle a greater variety of tasks, they do not respond with text that aligns with particular human-like characteristics. || We compare against four dialogue system baselines: Kvmemnn, Feed Yourself, Poly-encoder, and a BERT bi-ranker baseline trained on the Persona-Chat dataset using the same training hyperparameters (including learning rate scheduler and length capping settings) described in Section SECREF20. For the first three models, we use the provided pretrained (on Persona-Chat) models. We evaluate all four on our five evaluation characters discussed in Section SECREF28.`
- Baseline answer: `Pretrained models on Persona-Chat dataset (Kvmemnn, Feed Yourself, Poly-encoder, BERT bi-ranker)`
- Candidate answer: `Kvmemnn, Feed Yourself, Poly-encoder, BERT bi-ranker baseline`
- Baseline metrics: F1=0.7059, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.9231, support=1.0000, cite=0.5000
- Candidate cited context:

```text
[1] We compare against four dialogue system baselines: Kvmemnn, Feed Yourself, Poly-encoder, and a BERT bi-ranker baseline trained on the Persona-Chat dataset using the same training hyperparameters (including learning rate scheduler and length capping settings) described in Section SECREF20. For the first three models, we use the provided pretrained (on Persona-Chat) models. We evaluate all four on our five evaluation characters discussed in Section SECREF28.

[2] ALOHA, combined with the HLAs and dialogue dataset, achieves a significant improvement on the target character language style retrieval task compared to the baseline open-domain chatbot models. As observed from Table TABREF44, ALOHA achieves a significant boost in Hits@n/N accuracy and other metrics for retrieving the correct response of five diverse characters with different identities (see Section SECREF28).
```
- Candidate NLI claim summary:

```text
[1,2] supported=no entail=0.018 :: Kvmemnn, Feed Yourself, Poly-encoder, BERT bi-ranker baseline [1, 2]
```

## 10fb7dc031075946153baf0a0599e126de29e3a4 [citation_better]

- Question: How does the neural network architecture accomodate an unknown amount of senses per word?
- Gold answers: `converts WSD to a sequence learning task leverage gloss knowledge by extending gloss knowledge`
- Gold evidence: `The fourth block shows several recent neural-based methods. Bi-LSTM BIBREF11 is a baseline for neural models. Bi-LSTM$_{+ att. + LEX + POS}$ BIBREF10 is a multi-task learning framework for WSD, POS tagging, and LEX with self-attention mechanism, which converts WSD to a sequence learning task. GAS$_{ext}$ BIBREF12 is a variant of GAS which is a gloss-augmented variant of the memory network by extending gloss knowledge. CAN$^s$ and HCAN BIBREF13 are sentence-level and hierarchical co-attention neural network models which leverage gloss knowledge.`
- Baseline answer: `The neural network architecture treats WSD as a sentence-pair classification problem by constructing context-gloss pairs from all possible senses of the target word.`
- Candidate answer: `The neural network models leverage gloss knowledge by constructing context-gloss pairs from all possible senses of the target word, treating WSD as a sentence-pair classification problem.`
- Baseline metrics: F1=0.1212, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.2778, support=1.0000, cite=0.3333
- Candidate cited context:

```text
[2] Word Sense Disambiguation (WSD) is a fundamental task and long-standing challenge in Natural Language Processing (NLP), which aims to find the exact sense of an ambiguous word in a particular context BIBREF0. Previous WSD approaches can be grouped into two main categories: knowledge-based and supervised methods. Knowledge-based WSD methods rely on lexical resources like WordNet BIBREF1 and usually exploit two kinds of lexical knowledge. The gloss, which defines a word sense meaning, is first...

[3] Table TABREF19 shows the performance of our method on the English all-words WSD benchmark datasets. We compare our approach with previous methods. The first block shows the MFS baseline, which selects the most frequent sense in the training corpus for each target word. The second block shows two knowledge-based systems. Lesk$_{ext+emb}$ BIBREF4 is a variant of Lesk algorithm BIBREF2 by calculating the gloss-context overlap of the target word. Babelfy BIBREF6 is a unified graph-based approach...

[5] Since BIBREF17 map all the sense annotations in these datasets from their original versions to WordNet 3.0, we extract word sense glosses from WordNet 3.0.
```
- Candidate NLI claim summary:

```text
[2,3,5] supported=no entail=0.002 :: The neural network models leverage gloss knowledge by constructing context-gloss pairs from all possible senses of the target word, treating WSD as a sentence-pair classification problem [2, 3, 5].
```

## 692c9c5d9ff9cd3e0ce8b5e4fa68dda9bd23dec1 [citation_better]

- Question: How many users do they look at?
- Gold answers: `22,880 users || 20,000`
- Gold evidence: `The final set of categories is shown in Table TABREF1 , along with the number of users in each category. The resulting dataset consists of 22,880 users, 41,094 blogs, and 561,003 posts. Table TABREF2 presents additional statistics of our dataset. || Specifically, this paper makes four main contributions. First, we build a large, industry-annotated dataset that contains over 20,000 blog users. In addition to their posted text, we also link a number of user metadata including their gender, location, occupation, introduction and interests.`
- Baseline answer: `Over 20,000 blog users are looked at.`
- Candidate answer: `Over 20,000 blog users`
- Baseline metrics: F1=0.2500, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.4000, support=1.0000, cite=0.5000
- Candidate cited context:

```text
[3] Over the past two decades, the emergence of social media has enabled the proliferation of traceable human behavior. The content posted by users can reflect who their friends are, what topics they are interested in, or which company they are working for. At the same time, users are listing a number of profile fields to define themselves to others. The utilization of such metadata has proven important in facilitating further developments of applications in advertising BIBREF0 , personalization...

[4] Both of these recent studies are based on micro-blogging platforms, which inherently restrict the number of characters that a post can have, and consequently the way that users can express themselves. Moreover, both studies used off-the-shelf occupational taxonomies (rather than self-declared occupation categories), resulting in classes that are either too generic (e.g., media, welfare and electronic are three of the twelve Sina Weibo categories), or too intermixed (e.g., an assistant account...
```
- Candidate NLI claim summary:

```text
[3,4] supported=no entail=0.018 :: Over 20,000 blog users [3, 4]
```

## f318a2851d7061f05a5b32b94251f943480fbd15 [citation_better]

- Question: What conclusions do the authors draw from their finding that the emotional appeal of ISIS and Catholic materials are similar?
- Gold answers: `both corpuses used words that aim to inspire readers while avoiding fear actual words that lead to these effects are very different in the two contexts our findings indicate that, using proper methods, automated analysis of large bodies of textual data can provide novel insight insight into extremist propaganda || By comparing scores for each word calculated using Depechemood dictionary and normalize emotional score for each article, they found Catholic and ISIS materials show similar scores`
- Gold evidence: `Comparing these topics with those that appeared on a Catholic women forum, it seems that both ISIS and non-violent groups use topics about motherhood, spousal relationship, and marriage/divorce when they address women. Moreover, we used Depechemood methods to analyze the emotions that these materials are likely to elicit in readers. The result of our emotion analysis suggests that both corpuses used words that aim to inspire readers while avoiding fear. However, the actual words that lead to these effects are very different in the two contexts. Overall, our findings indicate that, using proper methods, automated analysis of large bodies of textual data can provide novel insight insight into extremist propaganda that can assist the counterterrorism community. || We rely on Depechemood dictionaries to analyze emotions in both corpora. These dictionaries are freely available and come in multiple arrangements. We used a version that includes words with their part of speech (POS) tags. Only words that exist in the Depechemood dictionary with the same POS tag are considered for our analysis. We aggregated the score for each word and normalized each article by emotions. To better compare the result, we added a baseline of 100 random articles from a Reuters news dataset as a non-religious general resource which is available in an NLTK python library. Figure FIGREF22 shows the aggregated score for different feelings in our corpora. || Both Catholic and ISIS related materials score the highest in “inspired” category. Furthermore, in both cases, being afraid has the lowest score. However, this is not the case for random news material such as the Reuters corpus, which are not that inspiring and, according to this method, seems to cause more fear in their audience. We investigate these results further by looking at the most inspiring words detected in these two corpora. Table TABREF24 presents 10 words that are among the most inspiring in both corpora. The comparison of the two lists indicate that the method picks very different words in each corpus to reach to the same conclusion. Also, we looked at separate articles in each of the issues of ISIS material addressing women. Figure FIGREF23 shows emotion scores in each of the 20 issues of ISIS propaganda. As demonstrated, in every separate article, this method gives the highest score to evoking inspirations in the reader. Also, in most of these issues the method scored “being afraid” as the lowest score in each issue.`
- Baseline answer: `Both ISIS and non-violent groups evoke similar emotions through topics about motherhood, spousal relationships, and marriage/divorce.`
- Candidate answer: `Both ISIS and non-violent groups evoke similar emotions related to motherhood, spousal relationships, and marriage/divorce in their materials targeting women.`
- Baseline metrics: F1=0.1905, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.2174, support=1.0000, cite=0.5000
- Candidate cited context:

```text
[3] In this section, we present the results of our analysis based on the contents of ISIS propaganda materials as compared to articles from the Catholic women forum. We then present the results of emotion analysis conducted on both corpora.

[4] In this paper, we have applied natural language processing methods to ISIS propaganda materials in an attempt to understand these materials using available technologies. We also compared these texts with a non-violent religious groups' (both focusing on women related articles) to examine possible similarities or differences in their approaches. To compare the contents, we used word frequency and topic modeling with NMF. Also, our results showed that NMF outperforms LDA due to the niche domain...
```
- Candidate NLI claim summary:

```text
[3,4] supported=no entail=0.001 :: Both ISIS and non-violent groups evoke similar emotions related to motherhood, spousal relationships, and marriage/divorce in their materials targeting women [3, 4].
```

## 02428a8fec9788f6dc3a86b5d5f3aa679935678d [baseline_better]

- Question: How do they define robustness of a model?
- Gold answers: `ability to accurately classify texts even when the amount of prior knowledge for different classes is unbalanced, and when the class distribution of the dataset is unbalanced || Low sensitivity to bias in prior knowledge`
- Gold evidence: `GE-FL reduces the heavy load of instance annotation and performs well when we provide prior knowledge with no bias. In our experiments, we observe that comparable numbers of labeled features for each class have to be supplied. But as mentioned before, it is often the case that we are not able to provide enough knowledge for some of the classes. For the baseball-hockey classification task, as shown before, GE-FL will predict most of the instances as baseball. In this section, we will show three terms to make the model more robust. || (a) We randomly select $t \in [1, 20]$ features from the feature pool for one class, and only one feature for the other. The original balanced movie dataset is used (positive:negative=1:1). || Our methods are also evaluated on datasets with different unbalanced class distributions. We manually construct several movie datasets with class distributions of 1:2, 1:3, 1:4 by randomly removing 50%, 67%, 75% positive documents. The original balanced movie dataset is used as a control group. We test with both balanced and unbalanced labeled features. For the balanced case, we randomly select 10 features from the feature pool for each class, and for the unbalanced case, we select 10 features for one class, and 1 feature for the other. Results are shown in Figure 3 . || Figure 3 (b) shows that when the labeled features are unbalanced, our methods significantly outperforms GE-FL. Incorporating KL divergence is robust enough to control unbalance both in the dataset and in labeled features while the other three methods are not so competitive. || However, a crucial problem, which has rarely been addressed, is the bias in the prior knowledge that we supply to the learning model. Would the model be robust or sensitive to the prior knowledge? Or, which kind of knowledge is appropriate for the task? Let's see an example: we may be a baseball fan but unfamiliar with hockey so that we can provide a few number of feature words of baseball, but much less of hockey for a baseball-hockey classification task. Such prior knowledge may mislead the model with heavy bias to baseball. If the model cannot handle this situation appropriately, the performance may be undesirable. || In this paper, we investigate into the problem in the framework of Generalized Expectation Criteria BIBREF7 . The study aims to reveal the factors of reducing the sensibility of the prior knowledge and therefore to make the model more robust and practical. To this end, we introduce auxiliary regularization terms in which our prior knowledge is formalized as distribution over output variables. Recall the example just mentioned, though we do not have enough knowledge to provide features for class hockey, it is easy for us to provide some neutral words, namely words that are not strong indicators of any class, like player here. As one of the factors revealed in this paper, supplying neutral feature words can boost the performance remarkably, making the model more robust.`
- Baseline answer: `The robustness of the model is justified by its insensitivity to the true distribution and ability to benefit from additional knowledge without heavy bias.`
- Candidate answer: `The robustness of a model is defined by its insensitivity to the true distribution, allowing for a rough estimation to be sufficient.`
- Baseline metrics: F1=0.3111, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.2381, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[2] This paper investigates into the problem of how to leverage prior knowledge robustly in learning models. We propose three regularization terms on top of generalized expectation criteria. As demonstrated by the experimental results, the performance can be considerably improved when taking into account these factors. Comparative results show that our proposed methods is more effective and works more robustly against baselines. To the best of our knowledge, this is the first work to address the...

[3] We address the robustness problem on top of GE-FL BIBREF0 , a GE method which leverages labeled features as prior knowledge. A labeled feature is a strong indicator of a specific class and is manually provided to the classifier. For example, words like amazing, exciting can be labeled features for class positive in sentiment classification.
```
- Candidate NLI claim summary:

```text
[2,3] supported=no entail=0.003 :: The robustness of a model is defined by its insensitivity to the true distribution, allowing for a rough estimation to be sufficient [2, 3].
```

## 50be4a737dc0951b35d139f51075011095d77f2a [baseline_better]

- Question: What background knowledge do they leverage?
- Gold answers: `labeled features || labelled features, which are words whose presence strongly indicates a specific class or topic`
- Gold evidence: `We address the robustness problem on top of GE-FL BIBREF0 , a GE method which leverages labeled features as prior knowledge. A labeled feature is a strong indicator of a specific class and is manually provided to the classifier. For example, words like amazing, exciting can be labeled features for class positive in sentiment classification. || As described in BIBREF0 , there are two ways to obtain labeled features. The first way is to use information gain. We first calculate the mutual information of all features according to the labels of the documents and select the top 20 as labeled features for each class as a feature pool. Note that using information gain requires the document label, but this is only to simulate how we human provide prior knowledge to the model. The second way is to use LDA BIBREF9 to select features. We use the same selection process as BIBREF0 , where they first train a LDA on the dataset, and then select the most probable features of each topic (sorted by $P(w_i|t_j)$ , the probability of word $w_i$ given topic $t_j$ ). || We evaluate our methods on several commonly used datasets whose themes range from sentiment, web-page, science to medical and healthcare. We use bag-of-words feature and remove stopwords in the preprocess stage. Though we have labels of all documents, we do not use them during the learning process, instead, we use the label of features.`
- Baseline answer: `Labeled features such as words indicating positive sentiment (e.g., amazing, exciting)`
- Candidate answer: `Leveraging labeled features such as words indicating positive sentiment (e.g., amazing, exciting) as prior knowledge in models.`
- Baseline metrics: F1=0.3077, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.2105, support=1.0000, cite=0.5000
- Candidate cited context:

```text
[3] We address the robustness problem on top of GE-FL BIBREF0 , a GE method which leverages labeled features as prior knowledge. A labeled feature is a strong indicator of a specific class and is manually provided to the classifier. For example, words like amazing, exciting can be labeled features for class positive in sentiment classification.

[4] There have been much work that incorporate prior knowledge into learning, and two related lines are surveyed here. One is to use prior knowledge to label unlabeled instances and then apply a standard learning algorithm. The other is to constrain the model directly with prior knowledge. Liu et al.text manually labeled features which are highly predictive to unsupervised clustering assignments and use them to label unlabeled data. Chang et al.guiding proposed constraint driven learning. They fi...
```

## 435570723b37ee1f5898c1a34ef86a0b2e8701bb [baseline_better]

- Question: Which translation systems do they compare against?
- Gold answers: `hierarchical phrase-based system BIBREF29 appropriate additional baseline would be to mark translation rules with these indicator functions but without the scores, akin to identifying rules with phrases in them (Baseline + SegOn) || English-Spanish MT system`
- Gold evidence: `To test this hypothesis, we built an English-Spanish MT system using the cdec decoder BIBREF27 for the entire training pipeline (word alignments, phrase extraction, feature weight tuning, and decoding). Corpora from the WMT 2011 evaluation was used to build the translation and language models, and for tuning (on news-test2010) and evaluation (on news-test2011), with scoring done using BLEU BIBREF28 . The baseline is a hierarchical phrase-based system BIBREF29 with a 4-gram language model, with feature weights tuned using MIRA BIBREF30 . For features, each translation rule is decorated with two lexical and phrasal features corresponding to the forward INLINEFORM0 and backward INLINEFORM1 conditional log frequencies, along with the log joint frequency INLINEFORM2 , the log frequency of the source phrase INLINEFORM3 , and whether the phrase pair or the source phrase is a singleton. Weights for the language model, glue rule, and word penalty are also tuned. This setup (Baseline) achieves scores en par with the published WMT results. || We added the compositionality score as an additional feature, and also added two binary-valued features: the first indicates if the given translation rule has not been decorated with a compositionality score (either because it consists of non-terminals only or the lexical items in the translation rule are unigrams), and correspondingly the second feature indicates if the translation rule has been scored. Therefore, an appropriate additional baseline would be to mark translation rules with these indicator functions but without the scores, akin to identifying rules with phrases in them (Baseline + SegOn).`
- Baseline answer: `The context compares against unweighted additive baseline models from previous work.`
- Candidate answer: `The unweighted additive baseline and additive models from previous work [BIBREF5, BIBREF23, BIBREF24] are compared against.`
- Baseline metrics: F1=0.0500, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0444, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[2] For the phrase similarity task we first compare our concatenation and tensor models learned using INLINEFORM0 and INLINEFORM1 regularization to three baselines: [noitemsep] add: INLINEFORM0 mult1: INLINEFORM0 mult2: INLINEFORM0 Other additive models from previous work BIBREF5 , BIBREF23 , BIBREF24 that impose varying amounts of structural assumptions on the semantic interactions between word representations e.g., INLINEFORM0 or INLINEFORM1 are subsumed by our concatenation model. The regulari...

[3] In this work, we presented two new sources of information for compositionality modeling and scoring, paraphrase information and context. For modeling, we showed that the paraphrase-learned compositional representations performs as well on a phrase similarity task as the average human annotator. For scoring, the importance of context was shown through the comparison of context-independent and dependent models. Improvements by the context-dependent model on an extrinsic machine translation task...
```

## 0d34c0812f1e69ea33f76ca8c24c23b0415ebc8d [baseline_better]

- Question: Which hand-crafted features are combined with word2vec?
- Gold answers: `three hand-crafted polarity scores, which are minimum, mean, and maximum polarity scores || polarity scores, which are minimum, mean, and maximum polarity scores, from each review`
- Gold evidence: `After creating several embeddings as mentioned above, we create document (review or tweet) vectors. For each document, we sum all the vectors of words occurring in that document and take their average. In addition to it, we extract three hand-crafted polarity scores, which are minimum, mean, and maximum polarity scores, from each review. These polarity scores of words are computed as in (DISPLAY_FORM4). For example, if a review consists of five words, it would have five polarity scores and we utilise only three of these sentiment scores as mentioned. Lastly, we concatenate these three scores to the averaged word vector per review. || That is, each review is represented by the average word vector of its constituent word embeddings and three supervised scores. We then feed these inputs into the SVM approach. The flowchart of our framework is given in Figure FIGREF11. When combining the unsupervised features, which are word vectors created on a word basis, with supervised three scores extracted on a review basis, we have better state-of-the-art results.`
- Baseline answer: `Minimum, mean, and maximum polarity scores of the reviews`
- Candidate answer: `3-feats features`
- Baseline metrics: F1=0.6000, support=0.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[4] We evaluated our models on four corpora, which are the movie and the Twitter datasets in Turkish and English. All of the embeddings are learnt on four corpora separately. We have used the accuracy metric since all the datasets are completely or nearly completely balanced. We performed 10-fold cross-validation for both of the datasets. We used the approximate randomisation technique to test whether our results are statistically significant. Here, we tried to predict the labels of reviews and a...
```
- Candidate NLI claim summary:

```text
[4] supported=no entail=0.003 :: 3-feats features [4]
```

## e949b28f6d1f20e18e82742e04d68158415dc61e [baseline_better]

- Question: What participating systems had better results than ones authors submitted?
- Gold answers: `For SLC task : Ituorp, ProperGander and YMJA teams had better results. For FLC task: newspeak and Antiganda teams had better results.`
- Gold evidence: `FLOAT SELECTED: Table 2: Comparison of our system (MIC-CIS) with top-5 participants: Scores on Test set for SLC and FLC`
- Baseline answer: `The authors' submissions were ranked 3rd and 4th in FLC and SLC tasks, respectively, better than some participating systems.`
- Candidate answer: `Other systems ranked 2nd and 1st in FLC and SLC tasks, respectively.`
- Baseline metrics: F1=0.2564, support=0.0000, cite=0.0000
- Candidate metrics: F1=0.2424, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[2] This paper describes our system (MIC-CIS) details and results of participation in the fine-grained propaganda detection shared task 2019. To address the tasks of sentence (SLC) and fragment level (FLC) propaganda detection, we explore different neural architectures (e.g., CNN, LSTM-CRF and BERT) and extract linguistic (e.g., part-of-speech, named entity, readability, sentiment, emotion, etc.), layout and topical features. Specifically, we have designed multi-granularity and multi-tasking neur...
```
- Candidate NLI claim summary:

```text
[2] supported=no entail=0.000 :: Other systems ranked 2nd and 1st in FLC and SLC tasks, respectively [2, 2].
```

## a0543b4afda15ea47c1e623c7f00d4aaca045be0 [yes_no]

- Question: Can named entities in SnapCaptions be discontigious?
- Gold answers: `no || unanswerable`
- Gold evidence: `Task: given a caption and a paired image (if used), the goal is to label every token in a caption in BIO scheme (B: beginning, I: inside, O: outside) BIBREF27 . We report the performance of the following state-of-the-art NER models as baselines, as well as several configurations of our proposed approach to examine contributions of each component (W: word, C: char, V: visual).`
- Baseline answer: `yes`
- Candidate answer: `yes`
- Baseline metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[3] Social media with abundant user-generated posts provide a rich platform for understanding events, opinions and preferences of groups and individuals. These insights are primarily hidden in unstructured forms of social media posts, such as in free-form text or images without tags. Named entity recognition (NER), the task of recognizing named entities from free-form text, is thus a critical step for building structural information, allowing for its use in personalized assistance, recommendation...

[4] Table TABREF6 shows the NER performance on the Snap Captions dataset. We report both entity types recognition (PER, LOC, ORG, MISC) and named entity segmentation (named entity or not) results. Parameters: We tune the parameters of each model with the following search space (bold indicate the choice for our final model): character embeddings dimension: {25, 50, 100, 150, 200, 300}, word embeddings size: {25, 50, 100, 150, 200, 300}, LSTM hidden states: {25, 50, 100, 150, 200, 300}, and INLINEF...
```
- Candidate NLI claim summary:

```text
[3,4] supported=no entail=0.002 :: yes [3, 4]
```

## 2cd37743bcc7ea3bd405ce6d91e79e5339d7642e [yes_no]

- Question: Do they report results only on English data?
- Gold answers: `yes`
- Gold evidence: `As a main field of interest in the current study, we chose controversies in education. One distinguishing feature of educational topics is their breadth of sub-topics and points of view, as they attract researchers, practitioners, parents, students, or policy-makers. We assume that this diversity leads to the linguistic variability of the education topics and thus represents a challenge for NLP. In a cooperation with researchers from the German Institute for International Educational Research we identified the following current controversial topics in education in English-speaking countries: (1) homeschooling, (2) public versus private schools, (3) redshirting — intentionally delaying the entry of an age-eligible child into kindergarten, allowing their child more time to mature emotionally and physically BIBREF51 , (4) prayer in schools — whether prayer in schools should be allowed and taken as a part of education or banned completely, (5) single-sex education — single-sex classes (males and females separate) versus mixed-sex classes (“co-ed”), and (6) mainstreaming — including children with special needs into regular classes.`
- Baseline answer: `no`
- Candidate answer: `no`
- Baseline metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[3] Park.Cardie.2014 annotated about 10k sentences from 1,047 documents into four types of argument propositions with Cohen's INLINEFORM0 0.73 on 30% of the dataset. Only 7% of the sentences were found to be non-argumentative. Faulkner2014 used Amazon Mechanical Turk to annotate 8,179 sentences from student essays. Three annotators decided whether the given sentence offered reasons for or against the main prompt of the essay (or no reason at all; 66% of the sentences were found to be neutral and...

[4] In the following experiment, we focus on automatic identification of arguments in the discourse. Our approach is based on supervised and semi-supervised machine learning methods on the gold data Toulmin dataset introduced in section SECREF51 . An argument consists of different components (such as premises, backing, etc.) which are implicitly linked to the claim. In principle one document can contain multiple independent arguments. However, only 4% of the documents in our dataset contain argum...
```
- Candidate NLI claim summary:

```text
[3,4] supported=no entail=0.001 :: no [3, 4]
```

## 88bf368491f9613767f696f84b4bb1f5a7d7cb48 [yes_no]

- Question: Does the professional translation or the machine translation introduce the artifacts?
- Gold answers: `yes`
- Gold evidence: `In this paper, we have shown that both human and machine translation can alter superficial patterns in data, which requires reconsidering previous findings in cross-lingual transfer learning. Based on the gained insights, we have improved the state-of-the-art in XNLI for the Translate-Test and Zero-Shot approaches by a substantial margin. Finally, we have shown that the phenomenon is not specific to NLI but also affects QA, although it is less pronounced there thanks to the translation procedure used in the corresponding benchmarks. So as to facilitate similar studies in the future, we release our NLI dataset, which, unlike previous benchmarks, was annotated in a non-English language and human translated into English.`
- Baseline answer: `yes`
- Candidate answer: `yes`
- Baseline metrics: F1=1.0000, support=0.0000, cite=0.0000
- Candidate metrics: F1=1.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[1] Both human and machine translation play a central role in cross-lingual transfer learning: many multilingual datasets have been created through professional translation services, and using machine translation to translate either the test set or the training set is a widely used transfer technique. In this paper, we show that such translation process can introduce subtle artifacts that have a notable impact in existing cross-lingual models. For instance, in natural language inference, translat...

[4] While most NLP resources are English-specific, there have been several recent efforts to build multilingual benchmarks. One possibility is to collect and annotate data in multiple languages separately BIBREF0, but most existing datasets have been created through translation BIBREF1, BIBREF2. This approach has two desirable properties: it relies on existing professional translation services rather than requiring expertise in multiple languages, and it results in parallel evaluation sets that o...

[5] So as to understand whether our previous findings apply to other tasks besides NLI, we run additional experiments on QA. As shown in Table TABREF32, BT-FI and BT-ES do indeed outperform Orig for the Translate-Test approach on MLQA. The improvement is modest, but very consistent across different languages, models and runs. The results for MT-ES and MT-FI are less conclusive, presumably because mapping the answer spans across languages might introduce some noise. In contrast, we do not observe...
```
- Candidate NLI claim summary:

```text
[1,4,5] supported=no entail=0.002 :: yes [1, 4, 5]
```

## ce6a3ca102a5ee62e86fc7def3b20b1f10d1eb25 [yes_no]

- Question: Do they report results only on English data?
- Gold answers: `unanswerable`
- Baseline answer: `no`
- Candidate answer: `unanswerable`
- Baseline metrics: F1=0.0000, support=0.0000, cite=0.0000
- Candidate metrics: F1=1.0000, support=1.0000, cite=0.0000

## 7793805982354947ea9fc742411bec314a6998f6 [yes_no]

- Question: Are the annotations automatic or manually created?
- Gold answers: `Automatic || We performed the annotation with freely available tools for the Portuguese language.`
- Gold evidence: `We annotated the dataset at two levels: Part of Speech (POS) and syntax. We performed the annotation with freely available tools for the Portuguese language. For POS we added a simple POS, that is, only type of word, and a fine-grained POS, which is the type of word plus its morphological features. We used the LX Parser BIBREF14 , for the simple POS and the Portuguese morphological module of Freeling BIBREF15 , for detailed POS. Concerning syntactic annotations, we included constituency and dependency annotations. For constituency parsing, we used the LX Parser, and for dependency, the DepPattern toolkit BIBREF16 .`
- Baseline answer: `unanswerable`
- Candidate answer: `no`
- Baseline metrics: F1=0.0000, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.0000, support=1.0000, cite=1.0000
- Candidate cited context:

```text
[1] As demonstrated earlier, these learner corpora use different formats. COPLE2 is mainly codified in XML, although it gives the possibility of getting the student version of the essay in TXT format. PEAPL2 and Leiria corpus are compiled in TXT format. In both corpora, the TXT files contain the student version with special annotations from the transcription. For the NLI experiments we were interested in a clean txt version of the students' text, together with versions annotated at different ling...
```
- Candidate NLI claim summary:

```text
[1] supported=no entail=0.001 :: no [1]
```

## 06be47e2f50b902b05ebf1ff1c66051925f5c247 [yes_no]

- Question: Do they use number of votes as an indicator of preference?
- Gold answers: `no || yes`
- Gold evidence: `FLOAT SELECTED: Figure 1: NATO-Warsaw intra-bloc differences in votes and speeches.`
- Baseline answer: `yes`
- Candidate answer: `yes`
- Baseline metrics: F1=1.0000, support=0.0000, cite=0.0000
- Candidate metrics: F1=1.0000, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[1] The most widely used source for deriving preferences in IR is UN roll call data BIBREF10 . Voting behavior represent a valuable source of revealed preference information, comparable across states and over time. However, UN roll call votes tend to be a weak signal of underlying preferences in cases where states vote for ceremonial purposes, are constrained by agenda-setting power dynamics, or vote as cohorts to maximize their impact within the UN, such as with regional blocs BIBREF11 . Similar...

[4] We draw on a recently released corpus of state speeches delivered during the annual UN General Debate that provides the first dataset of textual output from states that is recorded at regular time-series intervals and includes a sample of all countries that deliver speeches BIBREF11 . There are limitations to both votes and speeches in the UN in deriving estimates of states' underlying preferences. However, it is not controversial to suggest that state speeches can valuably complement roll ca...
```
- Candidate NLI claim summary:

```text
[1,4] supported=no entail=0.001 :: yes [1, 4]
```

## 1ec152119cf756b16191b236c85522afeed11f59 [random_fill]

- Question: What experiments are proposed to test that upper layers produce context-specific embeddings?
- Gold answers: `They measure self-similarity, intra-sentence similarity and maximum explainable variance of the embeddings in the upper layers. || They plot the average cosine similarity between uniformly random words increases exponentially from layers 8 through 12. They plot the average self-similarity of uniformly randomly sampled words in each layer of BERT, ELMo, and GPT-2 and shown that the higher layer produces more context-specific embeddings. They plot that word representations in a sentence become more context-specific in upper layers, they drift away from one another.`
- Gold evidence: `We measure how contextual a word representation is using three different metrics: self-similarity, intra-sentence similarity, and maximum explainable variance. || Recall from Definition 1 that the self-similarity of a word, in a given layer of a given model, is the average cosine similarity between its representations in different contexts, adjusted for anisotropy. If the self-similarity is 1, then the representations are not context-specific at all; if the self-similarity is 0, that the representations are maximally context-specific. In Figure FIGREF24, we plot the average self-similarity of uniformly randomly sampled words in each layer of BERT, ELMo, and GPT-2. For example, the self-similarity is 1.0 in ELMo's input layer because representations in that layer are static character-level embeddings. || In all three models, the higher the layer, the lower the self-similarity is on average. In other words, the higher the layer, the more context-specific the contextualized representations. This finding makes intuitive sense. In image classification models, lower layers recognize more generic features such as edges while upper layers recognize more class-specific features BIBREF19. Similarly, upper layers of LSTMs trained on NLP tasks learn more task-specific representations BIBREF4. Therefore, it follows that upper layers of neural language models learn more context-specific representations, so as to predict the next word for a given context more accurately. Of all three models, representations in GPT-2 are the most context-specific, with those in GPT-2's last layer being almost maximally context-specific. || As seen in Figure FIGREF20, for GPT-2, the average cosine similarity between uniformly randomly words is roughly 0.6 in layers 2 through 8 but increases exponentially from layers 8 through 12. In fact, word representations in GPT-2's last layer are so anisotropic that any two words have on average an almost perfect cosine similarity! This pattern holds for BERT and ELMo as well, though there are exceptions: for example, the anisotropy in BERT's penultimate layer is much higher than in its final layer. || As word representations in a sentence become more context-specific in upper layers, they drift away from one another, although there are exceptions (see layer 12 in Figure FIGREF25). However, in all layers, the average similarity between words in the same sentence is still greater than the average similarity between randomly chosen words (i.e., the anisotropy baseline). This suggests a more nuanced contextualization than in ELMo, with BERT recognizing that although the surrounding sentence informs a word's meaning, two words in the same sentence do not necessarily have a similar meaning because they share the same context.`
- Baseline answer: `Experiments comparing the similarity of word representations in the same sentence across different layers of ELMo, BERT, and GPT-2.`
- Candidate answer: `Experiments proposed include analyzing the similarity of word representations in the same sentence across different layers of ELMo, BERT, and GPT-2 to test context-specificity.`
- Baseline metrics: F1=0.3226, support=1.0000, cite=0.0000
- Candidate metrics: F1=0.2892, support=0.0000, cite=0.0000
- Candidate cited context:

```text
[1] In this paper, we investigated how contextual contextualized word representations truly are. For one, we found that upper layers of ELMo, BERT, and GPT-2 produce more context-specific representations than lower layers. This increased context-specificity is always accompanied by increased anisotropy. However, context-specificity also manifests differently across the three models; the anisotropy-adjusted similarity between words in the same sentence is highest in ELMo but almost non-existent in...

[2] Replacing static word embeddings with contextualized word representations has yielded significant improvements on many NLP tasks. However, just how contextual are the contextualized representations produced by models such as ELMo and BERT? Are there infinitely many context-specific representations for each word, or are words essentially assigned one of a finite number of word-sense representations? For one, we find that the contextualized representations of all words are not isotropic in any...

[5] The application of deep learning methods to NLP is made possible by representing words as vectors in a low-dimensional continuous space. Traditionally, these word embeddings were static: each word had a single vector, regardless of context BIBREF0, BIBREF1. This posed several problems, most notably that all senses of a polysemous word had to share the same representation. More recent work, namely deep neural language models such as ELMo BIBREF2 and BERT BIBREF3, have successfully created cont...
```
- Candidate NLI claim summary:

```text
[1,2,5] supported=no entail=0.000 :: Experiments proposed include analyzing the similarity of word representations in the same sentence across different layers of ELMo, BERT, and GPT-2 to test context-specificity [1, 2, 5].
```
