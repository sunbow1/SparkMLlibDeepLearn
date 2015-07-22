package CNN

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV,
  axpy => brzAxpy,
  svd => brzSvd,
  accumulate => Accumulate,
  rot90 => Rot90,
  sum => Bsum
}
import breeze.numerics.{
  exp => Bexp,
  tanh => Btanh
}

import scala.collection.mutable.ArrayBuffer
import java.util.Random
import scala.math._

/**
 * types：网络层类别
 * outputmaps：特征map数量
 * kernelsize：卷积核k大小
 * k: 卷积核
 * b: 偏置
 * dk: 卷积核的偏导
 * db: 偏置的偏导
 * scale: pooling大小
 */
case class CNNLayers(
  types: String,
  outputmaps: Double,
  kernelsize: Double,
  scale: Double,
  k: Array[Array[BDM[Double]]],
  b: Array[Double],
  dk: Array[Array[BDM[Double]]],
  db: Array[Double]) extends Serializable

/**
 * CNN(convolution neural network)卷积神经网络
 */

class CNN(
  private var mapsize: BDM[Double],
  private var types: Array[String],
  private var layer: Int,
  private var onum: Int,
  private var outputmaps: Array[Double],
  private var kernelsize: Array[Double],
  private var scale: Array[Double],
  private var alpha: Double,
  private var batchsize: Double,
  private var numepochs: Double) extends Serializable with Logging {
//        var mapsize = new BDM(1, 2, Array(28.0, 28.0))
//        var types = Array("i", "c", "s", "c", "s")
//        var layer = 5
//        var onum = 10  
//        var outputmaps = Array(0.0, 6.0, 0.0, 12.0, 0.0)
//        var kernelsize = Array(0.0, 5.0, 0.0, 5.0, 0.0)
//        var scale = Array(0.0, 0.0, 2.0, 0.0, 2.0)
//        var alpha = 1.0
//        var batchsize = 50.0
//        var numepochs = 1.0

  def this() = this(new BDM(1, 2, Array(28.0, 28.0)),
    Array("i", "c", "s", "c", "s"), 5, 10,
    Array(0.0, 6.0, 0.0, 12.0, 0.0),
    Array(0.0, 5.0, 0.0, 5.0, 0.0),
    Array(0.0, 0.0, 2.0, 0.0, 2.0),
    1.0, 50.0, 1.0)

  /** 设置输入层大小. Default: [28, 28]. */
  def setMapsize(mapsize: BDM[Double]): this.type = {
    this.mapsize = mapsize
    this
  }

  /** 设置网络层类别. Default: [1"i", "c", "s", "c", "s"]. */
  def setTypes(types: Array[String]): this.type = {
    this.types = types
    this
  }

  /** 设置网络层数. Default: 5. */
  def setLayer(layer: Int): this.type = {
    this.layer = layer
    this
  }

  /** 设置输出维度. Default: 10. */
  def setOnum(onum: Int): this.type = {
    this.onum = onum
    this
  }

  /** 设置特征map数量. Default: [0.0, 6.0, 0.0, 12.0, 0.0]. */
  def setOutputmaps(outputmaps: Array[Double]): this.type = {
    this.outputmaps = outputmaps
    this
  }

  /** 设置卷积核k大小. Default: [0.0, 5.0, 0.0, 5.0, 0.0]. */
  def setKernelsize(kernelsize: Array[Double]): this.type = {
    this.kernelsize = kernelsize
    this
  }

  /** 设置scale大小. Default: [0.0, 0.0, 2.0, 0.0, 2.0]. */
  def setScale(scale: Array[Double]): this.type = {
    this.scale = scale
    this
  }

  /** 设置学习因子. Default: 1. */
  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }

  /** 设置迭代大小. Default: 50. */
  def setBatchsize(batchsize: Double): this.type = {
    this.batchsize = batchsize
    this
  }

  /** 设置迭代次数. Default: 1. */
  def setNumepochs(numepochs: Double): this.type = {
    this.numepochs = numepochs
    this
  }

  /** 卷积神经网络层参数初始化. */
  def CnnSetup: (Array[CNNLayers], BDM[Double], BDM[Double], Double) = {
    var inputmaps1 = 1.0
    var mapsize1 = mapsize
    var confinit = ArrayBuffer[CNNLayers]()
    for (l <- 0 to layer - 1) { // layer
      val type1 = types(l)
      val outputmap1 = outputmaps(l)
      val kernelsize1 = kernelsize(l)
      val scale1 = scale(l)
      val layersconf = if (type1 == "s") { // 每一层参数初始化
        mapsize1 = mapsize1 / scale1
        val b1 = Array.fill(inputmaps1.toInt)(0.0)
        val ki = Array(Array(BDM.zeros[Double](1, 1)))
        new CNNLayers(type1, outputmap1, kernelsize1, scale1, ki, b1, ki, b1)
      } else if (type1 == "c") {
        mapsize1 = mapsize1 - kernelsize1 + 1.0
        val fan_out = outputmap1 * math.pow(kernelsize1, 2)
        val fan_in = inputmaps1 * math.pow(kernelsize1, 2)
        val ki = ArrayBuffer[Array[BDM[Double]]]()
        for (i <- 0 to inputmaps1.toInt - 1) { // input map
          val kj = ArrayBuffer[BDM[Double]]()
          for (j <- 0 to outputmap1.toInt - 1) { // output map          
            val kk = (BDM.rand[Double](kernelsize1.toInt, kernelsize1.toInt) - 0.5) * 2.0 * sqrt(6.0 / (fan_in + fan_out))
            kj += kk
          }
          ki += kj.toArray
        }
        val b1 = Array.fill(outputmap1.toInt)(0.0)
        inputmaps1 = outputmap1
        new CNNLayers(type1, outputmap1, kernelsize1, scale1, ki.toArray, b1, ki.toArray, b1)
      } else {
        val ki = Array(Array(BDM.zeros[Double](1, 1)))
        val b1 = Array(0.0)
        new CNNLayers(type1, outputmap1, kernelsize1, scale1, ki, b1, ki, b1)
      }
      confinit += layersconf
    }
    val fvnum = mapsize1(0, 0) * mapsize1(0, 1) * inputmaps1
    val ffb = BDM.zeros[Double](onum, 1)
    val ffW = (BDM.rand[Double](onum, fvnum.toInt) - 0.5) * 2.0 * sqrt(6.0 / (onum + fvnum))
    (confinit.toArray, ffb, ffW, alpha)
  }

  /**
   * 运行卷积神经网络算法.
   */
  def CNNtrain(train_d: RDD[(BDM[Double], BDM[Double])], opts: Array[Double]): CNNModel = {
    val sc = train_d.sparkContext
    var initStartTime = System.currentTimeMillis()
    var initEndTime = System.currentTimeMillis()
    // 参数初始化配置
    var (cnn_layers, cnn_ffb, cnn_ffW, cnn_alpha) = CnnSetup
    // 样本数据划分：训练数据、交叉检验数据
    val validation = opts(2)
    val splitW1 = Array(1.0 - validation, validation)
    val train_split1 = train_d.randomSplit(splitW1, System.nanoTime())
    val train_t = train_split1(0)
    val train_v = train_split1(1)
    // m:训练样本的数量
    val m = train_t.count
    // 计算batch的数量
    val batchsize = opts(0).toInt
    val numepochs = opts(1).toInt
    val numbatches = (m / batchsize).toInt
    var rL = Array.fill(numepochs * numbatches.toInt)(0.0)
    var n = 0
    // numepochs是循环的次数 
    for (i <- 1 to numepochs) {
      initStartTime = System.currentTimeMillis()
      val splitW2 = Array.fill(numbatches)(1.0 / numbatches)
      // 根据分组权重，随机划分每组样本数据  
      for (l <- 1 to numbatches) {
        // 权重 
        val bc_cnn_layers = sc.broadcast(cnn_layers)
        val bc_cnn_ffb = sc.broadcast(cnn_ffb)
        val bc_cnn_ffW = sc.broadcast(cnn_ffW)

        // 样本划分
        val train_split2 = train_t.randomSplit(splitW2, System.nanoTime())
        val batch_xy1 = train_split2(l - 1)

        // CNNff是进行前向传播
        // net = cnnff(net, batch_x);
        val train_cnnff = CNN.CNNff(batch_xy1, bc_cnn_layers, bc_cnn_ffb, bc_cnn_ffW)

        // CNNbp是后向传播
        // net = cnnbp(net, batch_y);
        val train_cnnbp = CNN.CNNbp(train_cnnff, bc_cnn_layers, bc_cnn_ffb, bc_cnn_ffW)

        // 权重更新
        //  net = cnnapplygrads(net, opts); 
        val train_nnapplygrads = CNN.CNNapplygrads(train_cnnbp, bc_cnn_ffb, bc_cnn_ffW, cnn_alpha)
        cnn_ffW = train_nnapplygrads._1
        cnn_ffb = train_nnapplygrads._2
        cnn_layers = train_nnapplygrads._3

        // error and loss
        // 输出误差计算
        // net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
        val rdd_loss1 = train_cnnbp._1.map(f => f._5)
        val (loss2, counte) = rdd_loss1.treeAggregate((0.0, 0L))(
          seqOp = (c, v) => {
            // c: (e, count), v: (m)
            val e1 = c._1
            val e2 = (v :* v).sum
            val esum = e1 + e2
            (esum, c._2 + 1)
          },
          combOp = (c1, c2) => {
            // c: (e, count)
            val e1 = c1._1
            val e2 = c2._1
            val esum = e1 + e2
            (esum, c1._2 + c2._2)
          })
        val Loss = (loss2 / counte.toDouble) * 0.5
        if (n == 0) {
          rL(n) = Loss
        } else {
          rL(n) = 0.09 * rL(n - 1) + 0.01 * Loss
        }
        n = n + 1
      }
      initEndTime = System.currentTimeMillis()
      // 打印输出结果
      printf("epoch: numepochs = %d , Took = %d seconds; batch train mse = %f.\n", i, scala.math.ceil((initEndTime - initStartTime).toDouble / 1000).toLong, rL(n - 1))
    }
    // 计算训练误差及交叉检验误差
    // Full-batch train mse
    var loss_train_e = 0.0
    var loss_val_e = 0.0
    loss_train_e = CNN.CNNeval(train_t, sc.broadcast(cnn_layers), sc.broadcast(cnn_ffb), sc.broadcast(cnn_ffW))
    if (validation > 0) loss_val_e = CNN.CNNeval(train_v, sc.broadcast(cnn_layers), sc.broadcast(cnn_ffb), sc.broadcast(cnn_ffW))
    printf("epoch: Full-batch train mse = %f, val mse = %f.\n", loss_train_e, loss_val_e)
    new CNNModel(cnn_layers, cnn_ffW, cnn_ffb)
  }

}

/**
 * NN(neural network)
 */
object CNN extends Serializable {

  // Initialization mode names

  /**
   * sigm激活函数
   * X = 1./(1+exp(-P));
   */
  def sigm(matrix: BDM[Double]): BDM[Double] = {
    val s1 = 1.0 / (Bexp(matrix * (-1.0)) + 1.0)
    s1
  }

  /**
   * tanh激活函数
   * f=1.7159*tanh(2/3.*A);
   */
  def tanh_opt(matrix: BDM[Double]): BDM[Double] = {
    val s1 = Btanh(matrix * (2.0 / 3.0)) * 1.7159
    s1
  }

  /**
   * 克罗内克积
   *
   */
  def expand(a: BDM[Double], s: Array[Int]): BDM[Double] = {
    // val a = BDM((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    // val s = Array(3, 2)
    val sa = Array(a.rows, a.cols)
    var tt = new Array[Array[Int]](sa.length)
    for (ii <- sa.length - 1 to 0 by -1) {
      var h = BDV.zeros[Int](sa(ii) * s(ii))
      h(0 to sa(ii) * s(ii) - 1 by s(ii)) := 1
      tt(ii) = Accumulate(h).data
    }
    var b = BDM.zeros[Double](tt(0).length, tt(1).length)
    for (j1 <- 0 to b.rows - 1) {
      for (j2 <- 0 to b.cols - 1) {
        b(j1, j2) = a(tt(0)(j1) - 1, tt(1)(j2) - 1)
      }
    }
    b
  }

  /**
   * convn卷积计算
   */
  def convn(m0: BDM[Double], k0: BDM[Double], shape: String): BDM[Double] = {
    //val m0 = BDM((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0))
    //val k0 = BDM((1.0, 1.0), (0.0, 1.0))
    //val m0 = BDM((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    //val k0 = BDM((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))    
    val out1 = shape match {
      case "valid" =>
        val m1 = m0
        val k1 = k0.t
        val row1 = m1.rows - k1.rows + 1
        val col1 = m1.cols - k1.cols + 1
        var m2 = BDM.zeros[Double](row1, col1)
        for (i <- 0 to row1 - 1) {
          for (j <- 0 to col1 - 1) {
            val r1 = i
            val r2 = r1 + k1.rows - 1
            val c1 = j
            val c2 = c1 + k1.cols - 1
            val mi = m1(r1 to r2, c1 to c2)
            m2(i, j) = (mi :* k1).sum
          }
        }
        m2
      case "full" =>
        var m1 = BDM.zeros[Double](m0.rows + 2 * (k0.rows - 1), m0.cols + 2 * (k0.cols - 1))
        for (i <- 0 to m0.rows - 1) {
          for (j <- 0 to m0.cols - 1) {
            m1((k0.rows - 1) + i, (k0.cols - 1) + j) = m0(i, j)
          }
        }
        val k1 = Rot90(Rot90(k0))
        val row1 = m1.rows - k1.rows + 1
        val col1 = m1.cols - k1.cols + 1
        var m2 = BDM.zeros[Double](row1, col1)
        for (i <- 0 to row1 - 1) {
          for (j <- 0 to col1 - 1) {
            val r1 = i
            val r2 = r1 + k1.rows - 1
            val c1 = j
            val c2 = c1 + k1.cols - 1
            val mi = m1(r1 to r2, c1 to c2)
            m2(i, j) = (mi :* k1).sum
          }
        }
        m2
    }
    out1
  }

  /**
   * cnnff是进行前向传播
   * 计算神经网络中的每个节点的输出值;
   */
  def CNNff(
    batch_xy1: RDD[(BDM[Double], BDM[Double])],
    bc_cnn_layers: org.apache.spark.broadcast.Broadcast[Array[CNNLayers]],
    bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[BDM[Double]],
    bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[BDM[Double]]): RDD[(BDM[Double], Array[Array[BDM[Double]]], BDM[Double], BDM[Double])] = {
    // 第1层:a(1)=[x]
    val train_data1 = batch_xy1.map { f =>
      val lable = f._1
      val features = f._2
      val nna1 = Array(features)
      val nna = ArrayBuffer[Array[BDM[Double]]]()
      nna += nna1
      (lable, nna)
    }
    // 第2至n-1层计算
    val train_data2 = train_data1.map { f =>
      val lable = f._1
      val nn_a = f._2
      var inputmaps1 = 1.0
      val n = bc_cnn_layers.value.length
      // for each layer
      for (l <- 1 to n - 1) {
        val type1 = bc_cnn_layers.value(l).types
        val outputmap1 = bc_cnn_layers.value(l).outputmaps
        val kernelsize1 = bc_cnn_layers.value(l).kernelsize
        val scale1 = bc_cnn_layers.value(l).scale
        val k1 = bc_cnn_layers.value(l).k
        val b1 = bc_cnn_layers.value(l).b
        val nna1 = ArrayBuffer[BDM[Double]]()
        if (type1 == "c") {
          for (j <- 0 to outputmap1.toInt - 1) { // output map 
            // create temp output map
            var z = BDM.zeros[Double](nn_a(l - 1)(0).rows - kernelsize1.toInt + 1, nn_a(l - 1)(0).cols - kernelsize1.toInt + 1)
            for (i <- 0 to inputmaps1.toInt - 1) { // input map
              // convolve with corresponding kernel and add to temp output map
              // z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
              z = z + convn(nn_a(l - 1)(i), k1(i)(j), "valid")
            }
            // add bias, pass through nonlinearity
            // net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j})
            val nna0 = sigm(z + b1(j))
            nna1 += nna0
          }
          nn_a += nna1.toArray
          inputmaps1 = outputmap1
        } else if (type1 == "s") {
          for (j <- 0 to inputmaps1.toInt - 1) {
            // z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid'); replace with variable
            // net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            val z = convn(nn_a(l - 1)(j), BDM.ones[Double](scale1.toInt, scale1.toInt) / (scale1 * scale1), "valid")
            val zs1 = z(::, 0 to -1 by scale1.toInt).t + 0.0
            val zs2 = zs1(::, 0 to -1 by scale1.toInt).t + 0.0
            val nna0 = zs2
            nna1 += nna0
          }
          nn_a += nna1.toArray
        }
      }
      // concatenate all end layer feature maps into vector
      val nn_fv1 = ArrayBuffer[Double]()
      for (j <- 0 to nn_a(n - 1).length - 1) {
        nn_fv1 ++= nn_a(n - 1)(j).data
      }
      val nn_fv = new BDM[Double](nn_fv1.length, 1, nn_fv1.toArray)
      // feedforward into output perceptrons
      // net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
      val nn_o = sigm(bc_cnn_ffW.value * nn_fv + bc_cnn_ffb.value)
      (lable, nn_a.toArray, nn_fv, nn_o)
    }
    train_data2
  }

  /**
   * CNNbp是后向传播
   * 计算权重的平均偏导数
   */
  def CNNbp(
    train_cnnff: RDD[(BDM[Double], Array[Array[BDM[Double]]], BDM[Double], BDM[Double])],
    bc_cnn_layers: org.apache.spark.broadcast.Broadcast[Array[CNNLayers]],
    bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[BDM[Double]],
    bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[BDM[Double]]): (RDD[(BDM[Double], Array[Array[BDM[Double]]], BDM[Double], BDM[Double], BDM[Double], BDM[Double], BDM[Double], Array[Array[BDM[Double]]])], BDM[Double], BDM[Double], Array[CNNLayers]) = {
    // error : net.e = net.o - y
    val n = bc_cnn_layers.value.length
    val train_data3 = train_cnnff.map { f =>
      val nn_e = f._4 - f._1
      (f._1, f._2, f._3, f._4, nn_e)
    }
    // backprop deltas
    // 输出层的 灵敏度 或者 残差
    // net.od = net.e .* (net.o .* (1 - net.o))
    // net.fvd = (net.ffW' * net.od)
    val train_data4 = train_data3.map { f =>
      val nn_e = f._5
      val nn_o = f._4
      val nn_fv = f._3
      val nn_od = nn_e :* (nn_o :* (1.0 - nn_o))
      val nn_fvd = if (bc_cnn_layers.value(n - 1).types == "c") {
        // net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
        val nn_fvd1 = bc_cnn_ffW.value.t * nn_od
        val nn_fvd2 = nn_fvd1 :* (nn_fv :* (1.0 - nn_fv))
        nn_fvd2
      } else {
        val nn_fvd1 = bc_cnn_ffW.value.t * nn_od
        nn_fvd1
      }
      (f._1, f._2, f._3, f._4, f._5, nn_od, nn_fvd)
    }
    // reshape feature vector deltas into output map style
    val sa1 = train_data4.map(f => f._2(n - 1)(1)).take(1)(0).rows
    val sa2 = train_data4.map(f => f._2(n - 1)(1)).take(1)(0).cols
    val sa3 = 1
    val fvnum = sa1 * sa2

    val train_data5 = train_data4.map { f =>
      val nn_a = f._2
      val nn_fvd = f._7
      val nn_od = f._6
      val nn_fv = f._3
      var nnd = new Array[Array[BDM[Double]]](n)
      val nnd1 = ArrayBuffer[BDM[Double]]()
      for (j <- 0 to nn_a(n - 1).length - 1) {
        val tmp1 = nn_fvd((j * fvnum) to ((j + 1) * fvnum - 1), 0)
        val tmp2 = new BDM(sa1, sa2, tmp1.data)
        nnd1 += tmp2
      }
      nnd(n - 1) = nnd1.toArray
      for (l <- (n - 2) to 0 by -1) {
        val type1 = bc_cnn_layers.value(l).types
        var nnd2 = ArrayBuffer[BDM[Double]]()
        if (type1 == "c") {
          for (j <- 0 to nn_a(l).length - 1) {
            val tmp_a = nn_a(l)(j)
            val tmp_d = nnd(l + 1)(j)
            val tmp_scale = bc_cnn_layers.value(l + 1).scale.toInt
            val tmp1 = tmp_a :* (1.0 - tmp_a)
            val tmp2 = expand(tmp_d, Array(tmp_scale, tmp_scale)) / (tmp_scale.toDouble * tmp_scale)
            nnd2 += (tmp1 :* tmp2)
          }
        } else if (type1 == "s") {
          for (i <- 0 to nn_a(l).length - 1) {
            var z = BDM.zeros[Double](nn_a(l)(0).rows, nn_a(l)(0).cols)
            for (j <- 0 to nn_a(l + 1).length - 1) {
              // z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
              z = z + convn(nnd(l + 1)(j), Rot90(Rot90(bc_cnn_layers.value(l + 1).k(i)(j))), "full")
            }
            nnd2 += z
          }
        }
        nnd(l) = nnd2.toArray
      }
      (f._1, f._2, f._3, f._4, f._5, f._6, f._7, nnd)
    }
    // dk db calc gradients
    var cnn_layers = bc_cnn_layers.value
    for (l <- 1 to n - 1) {
      val type1 = bc_cnn_layers.value(l).types
      val lena1 = train_data5.map(f => f._2(l).length).take(1)(0)
      val lena2 = train_data5.map(f => f._2(l - 1).length).take(1)(0)
      if (type1 == "c") {
        for (j <- 0 to lena1 - 1) {
          for (i <- 0 to lena2 - 1) {
            val rdd_dk_ij = train_data5.map { f =>
              val nn_a = f._2
              val nn_d = f._8
              val tmp_d = nn_d(l)(j)
              val tmp_a = nn_a(l - 1)(i)
              convn(Rot90(Rot90(tmp_a)), tmp_d, "valid")
            }
            val initdk = BDM.zeros[Double](rdd_dk_ij.take(1)(0).rows, rdd_dk_ij.take(1)(0).cols)
            val (dk_ij, count_dk) = rdd_dk_ij.treeAggregate((initdk, 0L))(
              seqOp = (c, v) => {
                // c: (m, count), v: (m)
                val m1 = c._1
                val m2 = m1 + v
                (m2, c._2 + 1)
              },
              combOp = (c1, c2) => {
                // c: (m, count)
                val m1 = c1._1
                val m2 = c2._1
                val m3 = m1 + m2
                (m3, c1._2 + c2._2)
              })
            val dk = dk_ij / count_dk.toDouble
            cnn_layers(l).dk(i)(j) = dk
          }
          val rdd_db_j = train_data5.map { f =>
            val nn_d = f._8
            val tmp_d = nn_d(l)(j)
            Bsum(tmp_d)
          }
          val db_j = rdd_db_j.reduce(_ + _)
          val count_db = rdd_db_j.count
          val db = db_j / count_db.toDouble
          cnn_layers(l).db(j) = db
        }
      }
    }

    // net.dffW = net.od * (net.fv)' / size(net.od, 2);
    // net.dffb = mean(net.od, 2);
    val train_data6 = train_data5.map { f =>
      val nn_od = f._6
      val nn_fv = f._3
      nn_od * nn_fv.t
    }
    val train_data7 = train_data5.map { f =>
      val nn_od = f._6
      nn_od
    }
    val initffW = BDM.zeros[Double](bc_cnn_ffW.value.rows, bc_cnn_ffW.value.cols)
    val (ffw2, countfffw2) = train_data6.treeAggregate((initffW, 0L))(
      seqOp = (c, v) => {
        // c: (m, count), v: (m)
        val m1 = c._1
        val m2 = m1 + v
        (m2, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (m, count)
        val m1 = c1._1
        val m2 = c2._1
        val m3 = m1 + m2
        (m3, c1._2 + c2._2)
      })
    val cnn_dffw = ffw2 / countfffw2.toDouble
    val initffb = BDM.zeros[Double](bc_cnn_ffb.value.rows, bc_cnn_ffb.value.cols)
    val (ffb2, countfffb2) = train_data7.treeAggregate((initffb, 0L))(
      seqOp = (c, v) => {
        // c: (m, count), v: (m)
        val m1 = c._1
        val m2 = m1 + v
        (m2, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (m, count)
        val m1 = c1._1
        val m2 = c2._1
        val m3 = m1 + m2
        (m3, c1._2 + c2._2)
      })
    val cnn_dffb = ffb2 / countfffb2.toDouble
    (train_data5, cnn_dffw, cnn_dffb, cnn_layers)
  }

  /**
   * NNapplygrads是权重更新
   * 权重更新
   */
  def CNNapplygrads(
    train_cnnbp: (RDD[(BDM[Double], Array[Array[BDM[Double]]], BDM[Double], BDM[Double], BDM[Double], BDM[Double], BDM[Double], Array[Array[BDM[Double]]])], BDM[Double], BDM[Double], Array[CNNLayers]),
    bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[BDM[Double]],
    bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[BDM[Double]],
    alpha: Double): (BDM[Double], BDM[Double], Array[CNNLayers]) = {
    val train_data5 = train_cnnbp._1
    val cnn_dffw = train_cnnbp._2
    val cnn_dffb = train_cnnbp._3
    var cnn_layers = train_cnnbp._4
    var cnn_ffb = bc_cnn_ffb.value
    var cnn_ffW = bc_cnn_ffW.value
    val n = cnn_layers.length

    for (l <- 1 to n - 1) {
      val type1 = cnn_layers(l).types
      val lena1 = train_data5.map(f => f._2(l).length).take(1)(0)
      val lena2 = train_data5.map(f => f._2(l - 1).length).take(1)(0)
      if (type1 == "c") {
        for (j <- 0 to lena1 - 1) {
          for (ii <- 0 to lena2 - 1) {
            cnn_layers(l).k(ii)(j) = cnn_layers(l).k(ii)(j) - cnn_layers(l).dk(ii)(j)
          }
          cnn_layers(l).b(j) = cnn_layers(l).b(j) - cnn_layers(l).db(j)
        }
      }
    }
    cnn_ffW = cnn_ffW + cnn_dffw
    cnn_ffb = cnn_ffb + cnn_dffb
    (cnn_ffW, cnn_ffb, cnn_layers)
  }

  /**
   * nneval是进行前向传播并计算输出误差
   * 计算神经网络中的每个节点的输出值，并计算平均误差;
   */
  def CNNeval(
    batch_xy1: RDD[(BDM[Double], BDM[Double])],
    bc_cnn_layers: org.apache.spark.broadcast.Broadcast[Array[CNNLayers]],
    bc_cnn_ffb: org.apache.spark.broadcast.Broadcast[BDM[Double]],
    bc_cnn_ffW: org.apache.spark.broadcast.Broadcast[BDM[Double]]): Double = {
    // CNNff是进行前向传播    
    val train_cnnff = CNN.CNNff(batch_xy1, bc_cnn_layers, bc_cnn_ffb, bc_cnn_ffW)
    // error and loss
    // 输出误差计算
    val rdd_loss1 = train_cnnff.map { f =>
      val nn_e = f._4 - f._1
      nn_e
    }
    val (loss2, counte) = rdd_loss1.treeAggregate((0.0, 0L))(
      seqOp = (c, v) => {
        // c: (e, count), v: (m)
        val e1 = c._1
        val e2 = (v :* v).sum
        val esum = e1 + e2
        (esum, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (e, count)
        val e1 = c1._1
        val e2 = c2._1
        val esum = e1 + e2
        (esum, c1._2 + c2._2)
      })
    val Loss = (loss2 / counte.toDouble) * 0.5
    Loss
  }
}

