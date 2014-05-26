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
        
        // Bias prepended to independent variable (city population)
        val X:List[List[Double]] = data map(_ match {
			case x :: xs => 1.0 :: List[Double](x)
			case _ => throw new NoSuchElementException
        })
        println("X: " + X)
        
        // target: franchise profit
        val y = data.map(_ drop 1).flatten
        println("y: " + y)
        
        // size of data
        val m = y.length
        println("m: " + m)
        
        val theta = List(List(0.0), List(0.0))
        
        //val iterations = 1500 // regression cycles
        val iterations = 15 // diagnostic
        val alpha = 0.01 // gradient step size
        
        // Initial cost calc
        val J = computeCost(X, y, theta)
        println("J: " + J)
        
        // theta calc via gradient descent
        val result:((Double, Double), List[Double]) = gradientDescent(X, y, theta, alpha, iterations)
        println("main - result: " + result)
    }
    
    def computeCost(X: List[List[Double]], y: List[Double], theta: List[List[Double]]): Double = {
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
		): ((Double, Double), List[Double]) = {
        
        val m = y.length
        val x = X map (x => x(1))
        
        //errorsProdXSum = sum((X * theta - y) .* x)
        val errorsProdXSum = dotProd(diff(matxProd(X, theta).flatten, y), x)
                
        def iterGradDesc(iter: Int) {
            if(!(iter > iterations)) { // recursion/iteration bounds
                                
                val errorsSum = diff(matxProd(X, theta).flatten, y).reduceLeft(_+_)
                val thetaTemp_1 = (theta(0)(0)) - (alpha * (1.0/m) * errorsSum)
                println("thetaTemp_1: " + thetaTemp_1)
                
                val errorsProdXSum = dotProd(diff(matxProd(X, theta).flatten, y), x)
                //theta_temp(2) = theta(2) - alpha * (1/m) * errorsProdXSum
                val thetaTemp_2 = (theta(1)(0)) - (alpha * (1.0/m) * errorsProdXSum)
                println("thetaTemp_2: " + thetaTemp_2)
                
                iterGradDesc(iter + 1) // recursive iteration
            }
        }
        
        iterGradDesc(0)
        
        ((0.0, 0.0), List(0.0)) // STUB - NIX
    }

    def diagnostic(elem: Any) {
        println(elem)
    }
}






