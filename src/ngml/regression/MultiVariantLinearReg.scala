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
        
        def calcSigma(X: List[List[Double]], stdDevs: List[Double]): List[Double] = {
            
            
            List(0.0) // STUB
        }
        
        val means = calcColsMu(X)
        val variances = avgSqrDiffs(X, means)
        val stdDeviations = variances.map(v => sqrt(v))
        val sigma = calcSigma(X, stdDeviations)
        
        (List(List(0.0)), means, List(0.0)) // STUB
    }
}