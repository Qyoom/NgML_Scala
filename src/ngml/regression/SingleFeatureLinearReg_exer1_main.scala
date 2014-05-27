package ngml.regression

import scala.io.Source
import scala.math._
import ngml.regression.SingleFeatureLinearReg._
import ngml.matrixlike.MatrixOps._

object SingleFeatureLinearReg_exer1_main {
	
    def main(args: Array[String]): Unit = {
        
        /* Training set:
         * Franchise profit correlated to city population size */ 
        val file = Source.fromFile("./test_data/ex1data1.txt")
        val data = file.getLines.toList.map( line => 
            line.split(",").toList.map(str => str.toDouble))
        println("data: " + data)
        
        // Intersect prepended to single independent variable (city population)
        val X = data map(_ match {
			case x :: xs => 1.0 :: List[Double](x)
			case _ => throw new NoSuchElementException
        })
        println("X is input, i.e. city population (prepended with intersect): " + X)
        
        // target: franchise profit
        val y = data.map(_ drop 1).flatten
        println("y is target, i.e. profit: " + y)
        
        // size of data
        val m = y.length
        println("data size m: " + m)
        
        val theta = List(List(0.0), List(0.0)) // clunky, needs its own Matrix class
        
        val iterations = 1500 // regression cycles
        val alpha = 0.01 // gradient step size
        
        // Initial cost calc
        val J = computeCost(X, y, theta)
        println("Initial cost calculation, i.e. J: " + J)
        
        // theta (slope parameters) calc via gradient descent
        // Also J_history of cost function output
        val result:(List[List[Double]], List[Double]) = 
            gradientDescent(X, y, theta, alpha, iterations)
            
        val finalTheta = result._1
        val J_history  = result._2
            
        println("gradient descent - theta final: " + finalTheta.flatten + 
                "\nJ_history (i.e. cost function): " + J_history)
                
        // Predict values for population size of 35,000
        val pop35 = List(List(1.0, 3.5))
        val predict1 = matxProd(pop35, finalTheta)
        println("Predicted profit for population size of 35,000: " + (predict1(0)(0)) * 10000)
        // Predict values for population size of 70,000
        val pop70 = List(List(1.0, 7.0))
        val predict2 = matxProd(pop70, finalTheta)
        println("Predicted profit for population size of 70,000: " + (predict2(0)(0)) * 10000)
    }
}