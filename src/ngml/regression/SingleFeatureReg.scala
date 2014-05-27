package ngml.regression

import scala.io.Source
import ngml.matrixlike.MatrixOps._
import scala.math._

/* Lab: Gradient descent, single variable with bias (1) */
object SingleFeatureReg {

    def main(args: Array[String]): Unit = {
        
        /* Training set:
         * Franchise profit correlated to city population size */ 
        val file = Source.fromFile("./test_data/ex1data1.txt")
        val data = file.getLines.toList.map( line => 
            line.split(",").toList.map(str => str.toDouble))
        println("data: " + data)
        
        // Intersection prepended to independent variable (city population)
        val X:List[List[Double]] = data map(_ match {
			case x :: xs => 1.0 :: List[Double](x)
			case _ => throw new NoSuchElementException
        })
        println("X (input, city population): " + X)
        
        // target: franchise profit
        val y = data.map(_ drop 1).flatten
        println("y (target, profit): " + y)
        
        // size of data
        val m = y.length
        println("data size - m: " + m)
        
        val theta = List(List(0.0), List(0.0))
        
        val iterations = 1500 // regression cycles
        val alpha = 0.01 // gradient step size
        
        // Initial cost calc
        val J = computeCost(X, y, theta)
        println("Initial cost calculation - J: " + J)
        
        // theta (slope parameters) calc via gradient descent
        // and J_history (cost function)
        val result:(List[List[Double]], List[Double]) = 
            gradientDescent(X, y, theta, alpha, iterations)
            
        val finalTheta = result._1
        val J_history = result._2
            
        println("gradient descent - theta final: " + finalTheta.flatten + 
                "\nJ_history (cost function): " + J_history)
                
        // Predict values for population size of 35,000
        val pop1 = List(List(1.0), List(3.5))
        val predict1 = matxProd(List(List(1.0, 3.5)), finalTheta)
        println("Predicted profit for population size of 35,000: " + (predict1(0)(0)) * 10000)
        // Predict values for population size of 70,000
        val predict2 = matxProd(List(List(1.0, 7.0)), finalTheta)
        println("Predicted profit for population size of 70,000: " + predict2(0)(0) * 10000)
    }
    
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
                
                iterGradDesc(iter + 1, newTheta, J :: J_history) // recursive iteration
            }
            else (theta, J_history) // final
        }
        
        // initiate gradient descent with empty theta and empty cost history (J)
        iterGradDesc(0, theta, List(0.0))
    }
}






