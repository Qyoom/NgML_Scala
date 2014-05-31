package ngml.regression

import scala.io.Source
import ngml.regression.MultiVariantLinearReg._

object MultiVariantLinearReg_exer1_main {

    def main(args: Array[String]): Unit = {
        
        /* Training set:
         * Predict house price correlated to sq. ft. and number of rooms */ 
        val file = Source.fromFile("./test_data/ex1data2.txt")
        val data = file.getLines.toList.map( line => 
            line.split(",").toList.map(str => str.toDouble))
        println("data: " + data)
        
        val X = data.map(x => x.take(2)) // matrix, i.e. List[List[Double]]
        println("X: " + X)
        
        val y = data.map(x => x.drop(2)).flatten // vector instead of matrix, i.e. List[Double]
        println("y: " + y)
        
        val m = y.length // scalar
        println("m: " + m)
        
        val normTup = featureNormalize(X)
        println("X_norm: " + normTup._1) // matrix
        println("mu: " + normTup._2) // vector
        println("sigma: " + normTup._3) // vector
        
        // Intersect prepended to single independent variables (sq. ft., number of rooms)
        val Xi = normTup._1.map(x => 1.0 :: x)
        println("X with intercept: " + Xi)
        
        val theta = List(List(0.0), List(0.0), List(0.0)) // clunky, needs its own Matrix class
        
        val iterations = 400 // regression cycles
        val alpha = 0.01 // gradient step size
        
        // Initial cost calc
        val J = computeCost(Xi, y, theta)
        println("Initial cost calculation, i.e. J: " + J)
        
        // theta (slope parameters) calc via gradient descent
        // Also J_history of cost function output
        val result:(List[List[Double]], List[Double]) = 
            gradientDescentMulti(Xi, y, theta, alpha, iterations)
            
        val finalTheta = result._1
        val J_history  = result._2
            
        println("Multivariable gradient descent - theta final: " + finalTheta.flatten + 
                "\nJ_history (i.e. cost function): " + J_history)
        
    } // end - main
}


