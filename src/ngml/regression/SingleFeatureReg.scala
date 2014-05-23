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
        
        val iterations = 1500 // regression cycles
        val alpha = 1.01 // gradient step size
        
        val J = computeCost(X, y, theta)
        println("J: " + J)
    }
    
    def computeCost(X: List[List[Double]], y: List[Double], theta: List[List[Double]]): Double = {
        val m = y.length
        val predictions = matxProd(X, theta).flatten
        val sqrErrors = dotPow(diff(predictions, y), 2)
        val sqrErrTot = sqrErrors.reduceLeft(_ + _)
        val mu = (1.0/(2*m))
        val J = mu * sqrErrTot
        J // cost
    }

}