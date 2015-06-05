package tests

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV,
  axpy => brzAxpy,
  svd => brzSvd,
  max => Bmax,
  min => Bmin,
  sum => Bsum
}
import scala.collection.mutable.ArrayBuffer
import DBN.DBN
import NN.NeuralNet
import util.RandSampleData

object Test_example_DBN {

  def main(args: Array[String]) {
    //1 构建Spark对象
    val conf = new SparkConf().setAppName("DBNtest")
    val sc = new SparkContext(conf)

    //*****************************例2（读取固定样本:来源于经典优化算法测试函数Sphere Model）*****************************// 
    //2 读取样本数据
    Logger.getRootLogger.setLevel(Level.WARN)
    val data_path = "/user/huangmeiling/deeplearn/data1"
    val examples = sc.textFile(data_path).cache()
    val train_d1 = examples.map { line =>
      val f1 = line.split("\t")
      val f = f1.map(f => f.toDouble)
      val id = f(0)
      val y = Array(f(1))
      val x = f.slice(2, f.length)
      (id, new BDM(1, y.length, y), new BDM(1, x.length, x))
    }
    val train_d = train_d1.map(f => (f._2, f._3))
    val opts = Array(100.0, 20.0, 0.0)

    //3 设置训练参数，建立DBN模型
    val DBNmodel = new DBN().
      setSize(Array(5, 7)).
      setLayer(2).
      setMomentum(0.1).
      setAlpha(1.0).
      DBNtrain(train_d, opts)

    //4 DBN模型转化为NN模型
    val mynn = DBNmodel.dbnunfoldtonn(1)
    val nnopts = Array(100.0, 50.0, 0.0)
    val numExamples = train_d.count()
    println(s"numExamples = $numExamples.")
    println(mynn._2)
    for (i <- 0 to mynn._1.length - 1) {
      print(mynn._1(i) + "\t")
    }
    println()
    println("mynn_W1")
    val tmpw1 = mynn._3(0)
    for (i <- 0 to tmpw1.rows - 1) {
      for (j <- 0 to tmpw1.cols - 1) {
        print(tmpw1(i, j) + "\t")
      }
      println()
    }
    val NNmodel = new NeuralNet().
      setSize(mynn._1).
      setLayer(mynn._2).
      setActivation_function("sigm").
      setOutput_function("sigm").
      setInitW(mynn._3).
      NNtrain(train_d, nnopts)

    //5 NN模型测试
    val NNforecast = NNmodel.predict(train_d)
    val NNerror = NNmodel.Loss(NNforecast)
    println(s"NNerror = $NNerror.")
    val printf1 = NNforecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)
    println("预测结果——实际值：预测值：误差")
    for (i <- 0 until printf1.length)
      println(printf1(i)._1 + "\t" + printf1(i)._2 + "\t" + (printf1(i)._2 - printf1(i)._1))

    //    //*****************************例3（读取图片数据，用户特征提取）*****************************// 
    //    //2 读取样本数据
    //    Logger.getRootLogger.setLevel(Level.WARN)
    //    val data_path = "/user/huangmeiling/deeplearn/train_d.txt"
    //    val examples = sc.textFile(data_path).cache()
    //    val train_d1 = examples.map { line =>
    //      val f1 = line.split("\t")
    //      val f = f1.map(f => f.toDouble)
    //      val y = f.slice(0, 9)
    //      val x = f.slice(10, f.length)
    //      (new BDM(1, y.length, y), (new BDM(1, x.length, x)) / 255.0)
    //    }
    //    val train_d = train_d1.map(f => (f._1, f._2))
    //    val opts = Array(100.0, 1.0, 0.0)
    //
    //    //3 设置训练参数，建立DBN模型
    //    val DBNmodel = new DBN().
    //      setSize(Array(784, 100, 100)).
    //      setLayer(3).
    //      setMomentum(0.0).
    //      setAlpha(1.0).
    //      DBNtrain(train_d, opts)

    //    //4 DBN模型转化为NN模型
    //    val mynn = DBNmodel.dbnunfoldtonn(1)
    //    val nnopts = Array(100.0, 50.0, 0.0)
    //    val numExamples = train_d.count()
    //    println(s"numExamples = $numExamples.")
    //    println(mynn._2)
    //    for (i <- 0 to mynn._1.length - 1) {
    //      print(mynn._1(i) + "\t")
    //    }
    //    println()
    //    println("mynn_W1")
    //    val tmpw1 = mynn._3(0)
    //    for (i <- 0 to tmpw1.rows - 1) {
    //      for (j <- 0 to tmpw1.cols - 1) {
    //        print(tmpw1(i, j) + "\t")
    //      }
    //      println()
    //    }
    //    val NNmodel = new NeuralNet().
    //      setSize(mynn._1).
    //      setLayer(mynn._2).
    //      setActivation_function("sigm").
    //      setOutput_function("sigm").
    //      setInitW(mynn._3).
    //      NNtrain(train_d, nnopts)
    //
    //    //5 NN模型测试
    //    val NNforecast = NNmodel.predict(train_d)
    //    val NNerror = NNmodel.Loss(NNforecast)
    //    println(s"NNerror = $NNerror.")
    //    val printf1 = NNforecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)
    //    println("预测结果——实际值：预测值：误差")
    //    for (i <- 0 until printf1.length)
    //      println(printf1(i)._1 + "\t" + printf1(i)._2 + "\t" + (printf1(i)._2 - printf1(i)._1))
  }
}