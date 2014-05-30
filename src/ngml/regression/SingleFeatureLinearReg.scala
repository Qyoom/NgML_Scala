package ngml.regression

import scala.io.Source
import ngml.matrixlike.MatrixOps._
import scala.math._

/* Lab: Gradient descent, single variable with intercept (1) */
object SingleFeatureLinearReg {

    def computeCost(X: List[List[Double]], y: List[Double], theta: List[List[Double]])
    		: Double = {
        val m = y.length
        val predictions = matxProd(X, theta).flatten
        val sqrErrors = dotPow(diff(predictions, y), 2)
        val sqrErrTotal = sqrErrors.reduceLeft(_ + _)
        val mu = (1.0/(2*m))
        val J = mu * sqrErrTotal
        J // cost
    }
    
    def gradientDescent(
        X: List[List[Double]],
        y: List[Double], 
        theta: List[List[Double]],
        alpha: Double, 
        iterations: Int
		): (List[List[Double]], List[Double]) = {
        
        val m = y.length
        val x = X map (x => x(1))
        
        // Returns final theta and history of cost function
        def iterGradDesc(iter: Int, theta: List[List[Double]], J_history: List[Double])
        		:(List[List[Double]], List[Double]) = {
            
            if(!(iter > iterations)) { // iteration bounds
                // This method looks at every example in the entire training 
                // set on every step, and is called batch gradient descent.
                val errorsSum = diff(matxProd(X, theta).flatten, y).reduceLeft(_+_)
                // update
                val newTheta_1 = theta(0)(0) - (alpha * (1.0/m) * errorsSum)
                
                val errorsProdXSum = dotProd(diff(matxProd(X, theta).flatten, y), x)
                // update
                val newTheta_2 = theta(1)(0) - (alpha * (1.0/m) * errorsProdXSum)
                
                val newTheta = List(List(newTheta_1), List(newTheta_2))
                val J = computeCost(X, y, newTheta)
                
                iterGradDesc(iter + 1, newTheta, J_history ++ List(J)) // recursive iteration
            }
            else (theta, J_history) // final
        }
        
        // initiate gradient descent with empty theta and empty cost history (J)
        iterGradDesc(0, theta, List(0.0))
    }
}






