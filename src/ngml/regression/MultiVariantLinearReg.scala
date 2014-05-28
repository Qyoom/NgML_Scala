package ngml.regression

import scala.math._

object MultiVariantLinearReg {

    // returns (X_norm, mu, sigma)
    def featureNormalize(X: List[List[Double]])
    	: (List[List[Double]], List[Double], List[Double]) = {
        
        // Recursive calc for List of mean of each column
        def calcColsMu(X: List[List[Double]]): List[Double] = {
            if(X.isEmpty) Nil
            else {
                val colMean = X.head.reduceRight(_+_) / X.length
                colMean :: calcColsMu(X.tail)
            }
        }
        
        // For X: The average of the squared differences from the Mean (mu)
        def avgSqrDiffs(X: List[List[Double]], mu: List[Double]): List[Double] = {
            if(X.isEmpty) Nil
            else {
                // for each number: subtract the Mean and square the result
                // (the squared difference)
                val sqrDiffs = X.head.map(x => pow((x - mu.head), 2))
                val avgSqrDiff = sqrDiffs.reduceRight(_+_) / X.length 
                avgSqrDiff :: avgSqrDiffs(X.tail, mu)
            }
        }
        
        def normalize(
            X: List[List[Double]],
            means: List[Double],
            stdDeviations: List[Double])
        	: List[List[Double]] = {
            
            if(X.isEmpty) Nil
            else {
                val normedVal = X.head.map(x => (x - means.head) / stdDeviations.head)
                normedVal :: normalize(X.tail, means, stdDeviations)
            }
        }
        
        val means = calcColsMu(X)
        val variances = avgSqrDiffs(X, means)
        val stdDeviations = variances.map(v => sqrt(v))
        val X_norm = normalize(X, means, stdDeviations)
        
        (X_norm, means, stdDeviations)
    } // end - featureNormalize
}