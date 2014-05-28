package ngml.regression

import scala.math._
import ngml.matrixlike.MatrixOps._

object MultiVariantLinearReg {

    // returns (X_norm, mu, sigma)
    def featureNormalize(X: List[List[Double]])
    	: (List[List[Double]], List[Double], List[Double]) = {
        
        // For X: The average of the squared differences from the Mean (mu)
        def calcAvgSqrDiffs(X: List[List[Double]], mu: List[Double]): List[Double] = {
            if(X.head.isEmpty) Nil
            else {
                // for each number: subtract the Mean and square the result
                // (the squared difference)
                val col = X map(_ head)                
                val diffs = col.map(x => x - mu.head)
                val sqrDiffs = diffs.map(x => pow(x, 2))
                val avgSqrDiffs = sqrDiffs.reduceRight(_+_) / (X.length - 1) // the -1 is important for training sets
                
                val otherCols = X map(_ tail)
                avgSqrDiffs :: calcAvgSqrDiffs(otherCols, mu.tail)
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
        
        val mu = sumCols(X) map(_ / X.length)
        val variances = calcAvgSqrDiffs(X, mu)
        val stdDeviations = variances.map(v => sqrt(v)) // square root of the Variance
        val X_norm = normalize(X, mu, stdDeviations)
        
        (X_norm, mu, stdDeviations)
    } // end - featureNormalize
}