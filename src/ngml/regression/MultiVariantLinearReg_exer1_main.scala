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
        
        val X = data.map(x => x.take(2))
        println("X: " + X)
        
        val y = data.map(x => x.drop(2))
        println("y: " + y)
        
        val m = y.length
        println("m: " + m)
        
        val normTup = featureNormalize(X)
        println("X_norm: " + normTup._1)
        println("mu: " + normTup._2)
        println("sigma: " + normTup._3)
        
        // Intersect prepended to single independent variables (sq. ft., number of rooms)
        val Xi = normTup._1.map(x => 1 :: x)
        println("X with intercept: " + Xi)
    } // end - main
}


