README.txt NLP, EXERCISE 2
===========================
AMIR FLEISHER
LUKAS ZBINDEN

------------------------------------------
CONTENTS OF ZIP FILE:
1) QUESTION 1: SEE q1.jpeg

2) MAIN SOURCE FILE: nlp_exercise_2.py

3) EXERCISE 2.C): PSEUDOCODE OF BIGRAM VITERBI ALGORITHM: SEE exercise_2_c_Viterbi_bigramHMM.jpg

4) EXERCISE 2. E): SEE confusion_matrix.txt

5) READ ME: this file README.txt
------------------------------------------
OUTPUT OF TEST RUN OF nlp_exercise_2.py, DECEMBER 10, 2017:

 --------------------- 
 ------- (b) --------- 
 --------------------- 
ErrorRates TEST_SET -->
0.08273219116321007 0.7893356643356644 0.16343849840255587
ErrorRates TEST_SET  <--
 --------------------- 
 ------- (c) --------- 
 --------------------- 
ErrorRates TEST_SET --> [ inital | (c) iii ]
0.6608844311022964 0.7615091607670845 0.7246654387908575
ErrorRates TEST_SET  <--
 --------------------- 
 ------- (d) --------- 
 --------------------- 
ErrorRates TEST_SET --> [ smoothing | (d) ii ]
0.23130537826515107 0.3341210497054653 0.2500924230274755
ErrorRates TEST_SET  <--
 --------------------- 
 ------- (e) --------- 
 --------------------- 
ErrorRates TEST_SET --> [ pseudo | (e) ii ]
0.7255516957046645 0.7888210204588787 0.7423794123206946
ErrorRates TEST_SET  <--
ErrorRates TEST_SET --> [ pseudo+smoothing | (e) iii ]
0.6836898976742436 0.7910590113395941 0.7161367362452596
ErrorRates TEST_SET  <--


CONCLUSION:
The Viterbi algorihm performs significantly better when add one smoothing is used. With only the 
psuedowords, the performance does not convince. The two techniques combined perform a little better
but by far not as good as smoothing alone. 


------------------------------------------
E) COMMENT ON MOST FREQUENT ERRORS:
The confusion matrix shows that the Viterbi using psuedowords and smoothing made the most prediction errors
with tags NN, IN and AT, the most frequent tags. 
