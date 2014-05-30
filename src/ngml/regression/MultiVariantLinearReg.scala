package ngml.regression

import scala.math._
import ngml.matrixlike.MatrixOps._

object MultiVariantLinearReg {

    // returns (X_norm, mu, sigma)
    def featureNormalize(X: List[List[Double]])
    	: (List[List[Double]], List[Double], List[Double]) = {
        
        // For X, The average of the squared differences from the Mean (mu)
        def calcAvgSqrDiffs(X: List[List[Double]], mu: List[Double]): List[Double] = {
            if(X.head.isEmpty) Nil
            else {
                // for each number: subtract the Mean and square the result
                // (the squared difference)
                val col = X map(_ head) // one col each iter             
                val diffs = col.map(x => x - mu.head) // diffs of each example in the col
                val sqrDiffs = diffs.map(x => pow(x, 2)) // square those individually
                val avgSqrDiff = sqrDiffs.reduceRight(_+_) / (X.length - 1) // One average for each col of the squared diffs. The -1 is important for training sets
                
                val otherCols = X map(_ tail)
                avgSqrDiff :: calcAvgSqrDiffs(otherCols, mu.tail) // collect the avgSqrDiffs, one per col 
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
    
    def computeCost(X: List[List[Double]], y: List[Double], theta: List[List[Double]])
    		: Double = {
        println("multi computeCost - size(X): " + size(X) + " y.length: " + y.length)
        val m = y.length
        val predictions = matxProd(X, theta).flatten
        println("predictions: " + predictions)
        val sqrErrors = dotPow(diff(predictions, y), 2)
        println("sqrErrors: " + sqrErrors)
        val sqrErrTotal = sqrErrors.reduceLeft(_ + _)
        println("sqrErrTotal: " + sqrErrTotal)
        val mu = (1.0/(2*m))
        val J = mu * sqrErrTotal
        J // cost
    }
    
    /*def gradientDescentMulti(
        X: List[List[Double]],
        y: List[Double], 
        theta: List[List[Double]],
        alpha: Double, 
        iterations: Int
		): (List[List[Double]], List[Double]) = {
        
        val m = y.length
        //val x = X map (x => x(1))
        
        // Returns final theta and history of cost function
        def iterGradDesc(iter: Int, theta: List[List[Double]], J_history: List[Double])
        		:(List[List[Double]], List[Double]) = {
            
            // theta = theta - alpha * (1/m) * (X' * (X * theta - y));
            if(!(iter > iterations)) { // iteration bounds
                
                val errors = diff(matxProd(X, theta).flatten, y)
                
                val errorsSum = errors.reduceLeft(_+_)
                val newTheta = theta - (alpha * (1.0/m) * errorsSum)
                
                val J = 0.0 // STUB
                iterGradDesc(iter + 1, newTheta, J :: J_history) // recursive iteration
            }
            else (theta, J_history) // final
        }
        
        // initiate gradient descent with empty theta and empty cost history (J)
        iterGradDesc(0, theta, List(0.0))
    }*/
}







