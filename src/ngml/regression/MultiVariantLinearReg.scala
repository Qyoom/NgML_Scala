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
        } // end - calcAvgSqrDiffs
        
        def normalize(
            X: List[List[Double]],
            means: List[Double],
            stdDeviations: List[Double])
        	: List[List[Double]] = {
            
            /*println("normalize TOP. size(X): " + size(X) + " means.length: " + 
                    means.length + " stdDeviations.length: " + stdDeviations.length)*/
            
            /* Octave version
             * X_norm(i,:) = (X_norm(i,:) - mu) ./ sigma;
             */
            
            if(X.isEmpty) Nil
            else {
                println("X.head: " + X.head)
                println("means: " + means)
                println("stdDeviations: " + stdDeviations)
                
                val aggTrans = List(X.head, means, stdDeviations).transpose
                println("aggTrans.length: " + aggTrans.length + " aggTrans: " + aggTrans)
                
                val normed = aggTrans.map(a => (a(0) - a(1)) / a(2))
                println("normed: " + normed)
                
                normed :: normalize(X.tail, means, stdDeviations)
            }
        } // end - normalize
        
        val mu = sumCols(X) map(_ / X.length)
        val variances = calcAvgSqrDiffs(X, mu)
        val stdDeviations = variances.map(v => sqrt(v)) // square root of the Variance
        val X_norm = normalize(X, mu, stdDeviations)
        
        (X_norm, mu, stdDeviations) // result
    } // end - featureNormalize
    
    def computeCost(X: List[List[Double]], y: List[Double], theta: List[List[Double]])
    		: Double = {
        val m = y.length
        val predictions = matxProd(X, theta).flatten
        //println("predictions: " + predictions)
        val sqrErrors = dotPow(diff(predictions, y), 2)
        //println("sqrErrors: " + sqrErrors)
        val sqrErrTotal = sqrErrors.reduceLeft(_ + _)
        //println("sqrErrTotal: " + sqrErrTotal)
        val mu = (1.0/(2*m))
        val J = mu * sqrErrTotal
        J // cost
    }
    
    def gradientDescentMulti(
        X: List[List[Double]],
        y: List[Double], 
        theta: List[List[Double]],
        alpha: Double, 
        iterations: Int
		): (List[List[Double]], List[Double]) = {
        
        val m = y.length 
        
        /*
        println("gradientDescentMulti - X: " + X)
        println("gradientDescentMulti - y: " + y)
        println("gradientDescentMulti - theta: " + theta)
        */

        // Returns final theta and history of cost function
        def iterGradDesc(iter: Int, theta: List[List[Double]], J_history: List[Double])
        		:(List[List[Double]], List[Double]) = {
            
            if(!(iter > iterations)) { // iteration bounds
                
                /* Octave version: theta = theta - alpha * (1/m) * (X' * (X * theta - y)); */
                
                val errorsVect = diff(matxProd(X, theta).flatten, y)
                //println("iterGradDesc - errorsVect: " + errorsVect)
                val errors = vectToMatrix(errorsVect) // TODO: internalize into a Matrix class
                val multXbyErrors = (matxProd(X.transpose, errors)).flatten
                val grad = alpha * (1.0/m)
                val deriv = multXbyErrors map(x => x * grad)
                val newTheta_vect = diff(theta.flatten, deriv)
                val newTheta = vectToMatrix(newTheta_vect)
                
                //println("iterGradDesc - newTheta: " + newTheta)
                
                val J = computeCost(X, y, newTheta)
                iterGradDesc(iter + 1, newTheta, J_history ++ List(J)) // recursive iteration
            }
            else (theta, J_history) // final
        }
        
        // initiate gradient descent with empty theta and empty cost history (J)
        iterGradDesc(0, theta, List(0.0))
        
    } // end - gradientDescentMulti
}







