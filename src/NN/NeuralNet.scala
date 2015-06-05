package NN

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
  svd => brzSvd
}
import breeze.numerics.{
  exp => Bexp,
  tanh => Btanh
}

import scala.collection.mutable.ArrayBuffer
import java.util.Random
import scala.math._

/**
 * label：目标矩阵
 * nna：神经网络每层节点的输出值,a(0),a(1),a(2)
 * error：输出层与目标值的误差矩阵
 */
case class NNLabel(label: BDM[Double], nna: ArrayBuffer[BDM[Double]], error: BDM[Double]) extends Serializable

/**
 * 配置参数
 */
case class NNConfig(
  size: Array[Int],
  layer: Int,
  activation_function: String,
  learningRate: Double,
  momentum: Double,
  scaling_learningRate: Double,
  weightPenaltyL2: Double,
  nonSparsityPenalty: Double,
  sparsityTarget: Double,
  inputZeroMaskedFraction: Double,
  dropoutFraction: Double,
  testing: Double,
  output_function: String) extends Serializable

/**
 * NN(neural network)
 */

class NeuralNet(
  private var size: Array[Int],
  private var layer: Int,
  private var activation_function: String,
  private var learningRate: Double,
  private var momentum: Double,
  private var scaling_learningRate: Double,
  private var weightPenaltyL2: Double,
  private var nonSparsityPenalty: Double,
  private var sparsityTarget: Double,
  private var inputZeroMaskedFraction: Double,
  private var dropoutFraction: Double,
  private var testing: Double,
  private var output_function: String,
  private var initW: Array[BDM[Double]]) extends Serializable with Logging {
  //          var size=Array(5, 7, 1)
  //          var layer=3
  //          var activation_function="tanh_opt"
  //          var learningRate=2.0
  //          var momentum=0.5
  //          var scaling_learningRate=1.0
  //          var weightPenaltyL2=0.0
  //          var nonSparsityPenalty=0.0
  //          var sparsityTarget=0.05
  //          var inputZeroMaskedFraction=0.0
  //          var dropoutFraction=0.0
  //          var testing=0.0
  //          var output_function="sigm"
  /**
   * size = architecture;
   * n = numel(nn.size);
   * activation_function = sigm   隐含层函数Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
   * learningRate = 2;            学习率learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
   * momentum = 0.5;              Momentum
   * scaling_learningRate = 1;    Scaling factor for the learning rate (each epoch)
   * weightPenaltyL2  = 0;        正则化L2 regularization
   * nonSparsityPenalty = 0;      权重稀疏度惩罚值on sparsity penalty
   * sparsityTarget = 0.05;       Sparsity target
   * inputZeroMaskedFraction = 0; 加入noise,Used for Denoising AutoEncoders
   * dropoutFraction = 0;         每一次mini-batch样本输入训练时，随机扔掉x%的隐含层节点Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
   * testing = 0;                 Internal variable. nntest sets this to one.
   * output = 'sigm';             输出函数output unit 'sigm' (=logistic), 'softmax' and 'linear'   *
   */
  def this() = this(NeuralNet.Architecture, 3, NeuralNet.Activation_Function, 2.0, 0.5, 1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, NeuralNet.Output, Array(BDM.zeros[Double](1, 1)))

  /** 设置神经网络结构. Default: [10, 5, 1]. */
  def setSize(size: Array[Int]): this.type = {
    this.size = size
    this
  }

  /** 设置神经网络层数据. Default: 3. */
  def setLayer(layer: Int): this.type = {
    this.layer = layer
    this
  }

  /** 设置隐含层函数. Default: sigm. */
  def setActivation_function(activation_function: String): this.type = {
    this.activation_function = activation_function
    this
  }

  /** 设置学习率因子. Default: 2. */
  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  /** 设置Momentum. Default: 0.5. */
  def setMomentum(momentum: Double): this.type = {
    this.momentum = momentum
    this
  }

  /** 设置scaling_learningRate. Default: 1. */
  def setScaling_learningRate(scaling_learningRate: Double): this.type = {
    this.scaling_learningRate = scaling_learningRate
    this
  }

  /** 设置正则化L2因子. Default: 0. */
  def setWeightPenaltyL2(weightPenaltyL2: Double): this.type = {
    this.weightPenaltyL2 = weightPenaltyL2
    this
  }

  /** 设置权重稀疏度惩罚因子. Default: 0. */
  def setNonSparsityPenalty(nonSparsityPenalty: Double): this.type = {
    this.nonSparsityPenalty = nonSparsityPenalty
    this
  }

  /** 设置权重稀疏度目标值. Default: 0.05. */
  def setSparsityTarget(sparsityTarget: Double): this.type = {
    this.sparsityTarget = sparsityTarget
    this
  }

  /** 设置权重加入噪声因子. Default: 0. */
  def setInputZeroMaskedFraction(inputZeroMaskedFraction: Double): this.type = {
    this.inputZeroMaskedFraction = inputZeroMaskedFraction
    this
  }

  /** 设置权重Dropout因子. Default: 0. */
  def setDropoutFraction(dropoutFraction: Double): this.type = {
    this.dropoutFraction = dropoutFraction
    this
  }

  /** 设置testing. Default: 0. */
  def setTesting(testing: Double): this.type = {
    this.testing = testing
    this
  }

  /** 设置输出函数. Default: linear. */
  def setOutput_function(output_function: String): this.type = {
    this.output_function = output_function
    this
  }

  /** 设置初始权重. Default: 0. */
  def setInitW(initW: Array[BDM[Double]]): this.type = {
    this.initW = initW
    this
  }

  /**
   * 运行神经网络算法.
   */
  def NNtrain(train_d: RDD[(BDM[Double], BDM[Double])], opts: Array[Double]): NeuralNetModel = {
    val sc = train_d.sparkContext
    var initStartTime = System.currentTimeMillis()
    var initEndTime = System.currentTimeMillis()
    // 参数配置 广播配置
    var nnconfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, testing,
      output_function)
    // 初始化权重
    var nn_W = NeuralNet.InitialWeight(size)
    if (!((initW.length == 1) && (initW(0) == (BDM.zeros[Double](1, 1))))) {
      for (i <- 0 to initW.length - 1) {
        nn_W(i) = initW(i)
      }
    }
    var nn_vW = NeuralNet.InitialWeightV(size)
    //        val tmpw = nn_W(1)
    //        for (i <- 0 to tmpw.rows -1) {
    //          for (j <- 0 to tmpw.cols - 1) {
    //            print(tmpw(i, j) + "\t")
    //          }
    //          println()
    //        }

    // 初始化每层的平均激活度nn.p
    // average activations (for use with sparsity)
    var nn_p = NeuralNet.InitialActiveP(size)

    // 样本数据划分：训练数据、交叉检验数据
    val validation = opts(2)
    val splitW1 = Array(1.0 - validation, validation)
    val train_split1 = train_d.randomSplit(splitW1, System.nanoTime())
    val train_t = train_split1(0)
    val train_v = train_split1(1)

    // m:训练样本的数量
    val m = train_t.count
    // batchsize是做batch gradient时候的大小 
    // 计算batch的数量
    val batchsize = opts(0).toInt
    val numepochs = opts(1).toInt
    val numbatches = (m / batchsize).toInt
    var L = Array.fill(numepochs * numbatches.toInt)(0.0)
    var n = 0
    var loss_train_e = Array.fill(numepochs)(0.0)
    var loss_val_e = Array.fill(numepochs)(0.0)
    // numepochs是循环的次数 
    for (i <- 1 to numepochs) {
      initStartTime = System.currentTimeMillis()
      val splitW2 = Array.fill(numbatches)(1.0 / numbatches)
      // 根据分组权重，随机划分每组样本数据  
      val bc_config = sc.broadcast(nnconfig)
      for (l <- 1 to numbatches) {
        // 权重 
        val bc_nn_W = sc.broadcast(nn_W)
        val bc_nn_vW = sc.broadcast(nn_vW)

        //        println(i + "\t" + l)
        //        val tmpw0 = bc_nn_W.value(0)
        //        for (i <- 0 to tmpw0.rows - 1) {
        //          for (j <- 0 to tmpw0.cols - 1) {
        //            print(tmpw0(i, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpw1 = bc_nn_W.value(1)
        //        for (i <- 0 to tmpw1.rows - 1) {
        //          for (j <- 0 to tmpw1.cols - 1) {
        //            print(tmpw1(i, j) + "\t")
        //          }
        //          println()
        //        }

        // 样本划分
        val train_split2 = train_t.randomSplit(splitW2, System.nanoTime())
        val batch_xy1 = train_split2(l - 1)
        //        val train_split3 = train_t.filter { f => (f._1 >= batchsize * (l - 1) + 1) && (f._1 <= batchsize * (l)) }
        //        val batch_xy1 = train_split3.map(f => (f._2, f._3))
        // Add noise to input (for use in denoising autoencoder)
        // 加入noise，这是denoising autoencoder需要使用到的部分  
        // 这部分请参见《Extracting and Composing Robust Features with Denoising Autoencoders》这篇论文  
        // 具体加入的方法就是把训练样例中的一些数据调整变为0，inputZeroMaskedFraction表示了调整的比例  
        //val randNoise = NeuralNet.RandMatrix(batch_x.numRows.toInt, batch_x.numCols.toInt, inputZeroMaskedFraction)
        val batch_xy2 = if (bc_config.value.inputZeroMaskedFraction != 0) {
          NeuralNet.AddNoise(batch_xy1, bc_config.value.inputZeroMaskedFraction)
        } else batch_xy1

        //        val tmpxy = batch_xy2.map(f => (f._1.toArray,f._2.toArray)).toArray.map {f => ((new ArrayBuffer() ++ f._1) ++ f._2).toArray}
        //        for (i <- 0 to tmpxy.length - 1) {
        //          for (j <- 0 to tmpxy(i).length - 1) {
        //            print(tmpxy(i)(j) + "\t")
        //          }
        //          println()
        //        }

        // NNff是进行前向传播
        // nn = nnff(nn, batch_x, batch_y);
        val train_nnff = NeuralNet.NNff(batch_xy2, bc_config, bc_nn_W)

        //        val tmpa0 = train_nnff.map(f => f._1.nna(0)).take(20)
        //        println("tmpa0")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpa0(i).cols - 1) {
        //            print(tmpa0(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpa1 = train_nnff.map(f => f._1.nna(1)).take(20)
        //        println("tmpa1")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpa1(i).cols - 1) {
        //            print(tmpa1(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpa2 = train_nnff.map(f => f._1.nna(2)).take(20)
        //        println("tmpa2")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpa2(i).cols - 1) {
        //            print(tmpa2(i)(0, j) + "\t")
        //          }
        //          println()
        //        }

        // sparsity计算，计算每层节点的平均稀疏度
        nn_p = NeuralNet.ActiveP(train_nnff, bc_config, nn_p)
        val bc_nn_p = sc.broadcast(nn_p)

        // NNbp是后向传播
        // nn = nnbp(nn);
        val train_nnbp = NeuralNet.NNbp(train_nnff, bc_config, bc_nn_W, bc_nn_p)

        //        val tmpd0 = rdd5.map(f => f._2(2)).take(20)
        //        println("tmpd0")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpd0(i).cols - 1) {
        //            print(tmpd0(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpd1 = rdd5.map(f => f._2(1)).take(20)
        //        println("tmpd1")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpd1(i).cols - 1) {
        //            print(tmpd1(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpdw0 = rdd5.map(f => f._3(0)).take(20)
        //        println("tmpdw0")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpdw0(i).cols - 1) {
        //            print(tmpdw0(i)(0, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpdw1 = rdd5.map(f => f._3(1)).take(20)
        //        println("tmpdw1")
        //        for (i <- 0 to 10) {
        //          for (j <- 0 to tmpdw1(i).cols - 1) {
        //            print(tmpdw1(i)(0, j) + "\t")
        //          }
        //          println()
        //        }

        // nn = NNapplygrads(nn) returns an neural network structure with updated
        // weights and biases
        // 更新权重参数：w=w-α*[dw + λw]    
        val train_nnapplygrads = NeuralNet.NNapplygrads(train_nnbp, bc_config, bc_nn_W, bc_nn_vW)
        nn_W = train_nnapplygrads(0)
        nn_vW = train_nnapplygrads(1)

        //        val tmpw2 = train_nnapplygrads(0)(0)
        //        for (i <- 0 to tmpw2.rows - 1) {
        //          for (j <- 0 to tmpw2.cols - 1) {
        //            print(tmpw2(i, j) + "\t")
        //          }
        //          println()
        //        }
        //        val tmpw3 = train_nnapplygrads(0)(1)
        //        for (i <- 0 to tmpw3.rows - 1) {
        //          for (j <- 0 to tmpw3.cols - 1) {
        //            print(tmpw3(i, j) + "\t")
        //          }
        //          println()
        //        }

        // error and loss
        // 输出误差计算
        val loss1 = train_nnff.map(f => f._1.error)
        val (loss2, counte) = loss1.treeAggregate((0.0, 0L))(
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
        val Loss = loss2 / counte.toDouble
        L(n) = Loss * 0.5
        n = n + 1
      }
      // 计算本次迭代的训练误差及交叉检验误差
      // Full-batch train mse
      val evalconfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
        weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, 1.0,
        output_function)
      loss_train_e(i - 1) = NeuralNet.NNeval(train_t, sc.broadcast(evalconfig), sc.broadcast(nn_W))
      if (validation > 0) loss_val_e(i - 1) = NeuralNet.NNeval(train_v, sc.broadcast(evalconfig), sc.broadcast(nn_W))

      // 更新学习因子
      // nn.learningRate = nn.learningRate * nn.scaling_learningRate;
      nnconfig = NNConfig(size, layer, activation_function, nnconfig.learningRate * nnconfig.scaling_learningRate, momentum, scaling_learningRate,
        weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, testing,
        output_function)
      initEndTime = System.currentTimeMillis()

      // 打印输出结果
      printf("epoch: numepochs = %d , Took = %d seconds; Full-batch train mse = %f, val mse = %f.\n", i, scala.math.ceil((initEndTime - initStartTime).toDouble / 1000).toLong, loss_train_e(i - 1), loss_val_e(i - 1))
    }
    val configok = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, 1.0,
      output_function)
    new NeuralNetModel(configok, nn_W)
  }

}

/**
 * NN(neural network)
 */
object NeuralNet extends Serializable {

  // Initialization mode names
  val Activation_Function = "sigm"
  val Output = "linear"
  val Architecture = Array(10, 5, 1)

  /**
   * 增加随机噪声
   * 若随机值>=Fraction，值不变，否则改为0
   */
  def AddNoise(rdd: RDD[(BDM[Double], BDM[Double])], Fraction: Double): RDD[(BDM[Double], BDM[Double])] = {
    val addNoise = rdd.map { f =>
      val features = f._2
      val a = BDM.rand[Double](features.rows, features.cols)
      val a1 = a :>= Fraction
      val d1 = a1.data.map { f => if (f == true) 1.0 else 0.0 }
      val a2 = new BDM(features.rows, features.cols, d1)
      val features2 = features :* a2
      (f._1, features2)
    }
    addNoise
  }

  /**
   * 初始化权重
   * 初始化为一个很小的、接近零的随机值
   */
  def InitialWeight2(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_W = ArrayBuffer[BDM[Double]]()
    val d1 = BDM((2.54631575950577, -2.72375471180638, -1.83131523622017, -0.832303531504013, -1.28869970471936, -0.460188104184124), (-1.52091024201213, 1.81815348316090, -0.533406209340414, 1.77153723107141, -1.70376378930231, 1.95852409868481), (0.604392922735100, -0.312805008341265, 2.46338861792203, -2.77264318419692, -2.74202474572555, 0.142284005609256), (-0.0792951314491902, 0.652983968878905, 2.35836765255640, -2.04274164893227, 1.39603060318734, -1.68208055847319), (2.21352121948139, 1.65144527075334, -0.507588360889342, -1.68141383648426, -0.310581480324221, 0.973756570035639), (1.48264358368951, 2.38613449604874, 2.22681802175890, -1.70428719030501, 2.44271213316363, 1.91268676272635), (-0.246256073282793, 1.34750367072394, -2.50094445126864, 0.587138926992906, -0.192365052800164, -2.71732925728203))
    nn_W += d1
    val d2 = BDM((1.25592501437006, -0.834980000207940, 2.29875024099543, 0.0194882319892158, 1.45126037957791, -0.492648144141757, -1.35365058999520, -2.15014190874756))
    nn_W += d2
    nn_W.toArray
  }
  def InitialWeight(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_W = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.rand(size(i), size(i - 1) + 1)
      d1 :-= 0.5
      val f1 = 2 * 4 * sqrt(6.0 / (size(i) + size(i - 1)))
      val d2 = d1 :* f1
      //val d3 = new DenseMatrix(d2.rows, d2.cols, d2.data, d2.isTranspose)
      //val d4 = Matrices.dense(d2.rows, d2.cols, d2.data)
      nn_W += d2
    }
    nn_W.toArray
  }

  /**
   * 初始化权重vW
   * 初始化为0
   */
  def InitialWeightV(size: Array[Int]): Array[BDM[Double]] = {
    // 初始化权重参数
    // weights and weight momentum
    // nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    val n = size.length
    val nn_vW = ArrayBuffer[BDM[Double]]()
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](size(i), size(i - 1) + 1)
      nn_vW += d1
    }
    nn_vW.toArray
  }

  /**
   * 初始每一层的平均激活度
   * 初始化为0
   */
  def InitialActiveP(size: Array[Int]): Array[BDM[Double]] = {
    // 初始每一层的平均激活度
    // average activations (for use with sparsity)
    // nn.p{i}     = zeros(1, nn.size(i));  
    val n = size.length
    val nn_p = ArrayBuffer[BDM[Double]]()
    nn_p += BDM.zeros[Double](1, 1)
    for (i <- 1 to n - 1) {
      val d1 = BDM.zeros[Double](1, size(i))
      nn_p += d1
    }
    nn_p.toArray
  }

  /**
   * 随机让网络某些隐含层节点的权重不工作
   * 若随机值>=Fraction，矩阵值不变，否则改为0
   */
  def DropoutWeight(matrix: BDM[Double], Fraction: Double): Array[BDM[Double]] = {
    val aa = BDM.rand[Double](matrix.rows, matrix.cols)
    val aa1 = aa :> Fraction
    val d1 = aa1.data.map { f => if (f == true) 1.0 else 0.0 }
    val aa2 = new BDM(matrix.rows: Int, matrix.cols: Int, d1: Array[Double])
    val matrix2 = matrix :* aa2
    Array(aa2, matrix2)
  }

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
   * nnff是进行前向传播
   * 计算神经网络中的每个节点的输出值;
   */
  def NNff(
    batch_xy2: RDD[(BDM[Double], BDM[Double])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): RDD[(NNLabel, Array[BDM[Double]])] = {
    // 第1层:a(1)=[1 x]
    // 增加偏置项b
    val train_data1 = batch_xy2.map { f =>
      val lable = f._1
      val features = f._2
      val nna = ArrayBuffer[BDM[Double]]()
      val Bm1 = new BDM(features.rows, 1, Array.fill(features.rows * 1)(1.0))
      val features2 = BDM.horzcat(Bm1, features)
      val error = BDM.zeros[Double](lable.rows, lable.cols)
      nna += features2
      NNLabel(lable, nna, error)
    }

    //    println("bc_size " + bc_config.value.size(0) + bc_config.value.size(1) + bc_config.value.size(2))
    //    println("bc_layer " + bc_config.value.layer)
    //    println("bc_activation_function " + bc_config.value.activation_function)
    //    println("bc_output_function " + bc_config.value.output_function)
    //
    //    println("tmpw0 ")
    //    val tmpw0 = bc_nn_W.value(0)
    //    for (i <- 0 to tmpw0.rows - 1) {
    //      for (j <- 0 to tmpw0.cols - 1) {
    //        print(tmpw0(i, j) + "\t")
    //      }
    //      println()
    //    }

    // feedforward pass
    // 第2至n-1层计算，a(i)=f(a(i-1)*w(i-1)')
    //val tmp1 = train_data1.map(f => f.nna(0).data).take(1)(0)
    //val tmp2 = new BDM(1, tmp1.length, tmp1)
    //val nn_a = ArrayBuffer[BDM[Double]]()
    //nn_a += tmp2
    val train_data2 = train_data1.map { f =>
      val nn_a = f.nna
      val dropOutMask = ArrayBuffer[BDM[Double]]()
      dropOutMask += new BDM[Double](1, 1, Array(0.0))
      for (j <- 1 to bc_config.value.layer - 2) {
        // 计算每层输出
        // Calculate the unit's outputs (including the bias term)
        // nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}')
        // nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');            
        val A1 = nn_a(j - 1)
        val W1 = bc_nn_W.value(j - 1)
        val aw1 = A1 * W1.t
        val nnai1 = bc_config.value.activation_function match {
          case "sigm" =>
            val aw2 = NeuralNet.sigm(aw1)
            aw2
          case "tanh_opt" =>
            val aw2 = NeuralNet.tanh_opt(aw1)
            //val aw2 = Btanh(aw1 * (2.0 / 3.0)) * 1.7159
            aw2
        }
        // dropout计算
        // Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分
        // 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        // 参照 http://www.cnblogs.com/tornadomeet/p/3258122.html   
        val dropoutai = if (bc_config.value.dropoutFraction > 0) {
          if (bc_config.value.testing == 1) {
            val nnai2 = nnai1 * (1.0 - bc_config.value.dropoutFraction)
            Array(new BDM[Double](1, 1, Array(0.0)), nnai2)
          } else {
            NeuralNet.DropoutWeight(nnai1, bc_config.value.dropoutFraction)
          }
        } else {
          val nnai2 = nnai1
          Array(new BDM[Double](1, 1, Array(0.0)), nnai2)
        }
        val nnai2 = dropoutai(1)
        dropOutMask += dropoutai(0)
        // Add the bias term
        // 增加偏置项b
        // nn.a{i} = [ones(m,1) nn.a{i}];
        val Bm1 = BDM.ones[Double](nnai2.rows, 1)
        val nnai3 = BDM.horzcat(Bm1, nnai2)
        nn_a += nnai3
      }
      (NNLabel(f.label, nn_a, f.error), dropOutMask.toArray)
    }

    // 输出层计算
    val train_data3 = train_data2.map { f =>
      val nn_a = f._1.nna
      // nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
      // nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';          
      val An1 = nn_a(bc_config.value.layer - 2)
      val Wn1 = bc_nn_W.value(bc_config.value.layer - 2)
      val awn1 = An1 * Wn1.t
      val nnan1 = bc_config.value.output_function match {
        case "sigm" =>
          val awn2 = NeuralNet.sigm(awn1)
          //val awn2 = 1.0 / (Bexp(awn1 * (-1.0)) + 1.0)
          awn2
        case "linear" =>
          val awn2 = awn1
          awn2
      }
      nn_a += nnan1
      (NNLabel(f._1.label, nn_a, f._1.error), f._2)
    }

    // error and loss
    // 输出误差计算
    // nn.e = y - nn.a{n};
    // val nn_e = batch_y - nnan
    val train_data4 = train_data3.map { f =>
      val batch_y = f._1.label
      val nnan = f._1.nna(bc_config.value.layer - 1)
      val error = (batch_y - nnan)
      (NNLabel(f._1.label, f._1.nna, error), f._2)
    }
    train_data4
  }

  /**
   * sparsity计算，网络稀疏度
   * 计算每个节点的平均值
   */
  def ActiveP(
    train_nnff: RDD[(NNLabel, Array[BDM[Double]])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    nn_p_old: Array[BDM[Double]]): Array[BDM[Double]] = {
    val nn_p = ArrayBuffer[BDM[Double]]()
    nn_p += BDM.zeros[Double](1, 1)
    // calculate running exponential activations for use with sparsity
    // sparsity计算，计算sparsity，nonSparsityPenalty 是对没达到sparsitytarget的参数的惩罚系数 
    for (i <- 1 to bc_config.value.layer - 1) {
      val pi1 = train_nnff.map(f => f._1.nna(i))
      val initpi = BDM.zeros[Double](1, bc_config.value.size(i))
      val (piSum, miniBatchSize) = pi1.treeAggregate((initpi, 0L))(
        seqOp = (c, v) => {
          // c: (nnasum, count), v: (nna)
          val nna1 = c._1
          val nna2 = v
          val nnasum = nna1 + nna2
          (nnasum, c._2 + 1)
        },
        combOp = (c1, c2) => {
          // c: (nnasum, count)
          val nna1 = c1._1
          val nna2 = c2._1
          val nnasum = nna1 + nna2
          (nnasum, c1._2 + c2._2)
        })
      val piAvg = piSum / miniBatchSize.toDouble
      val oldpi = nn_p_old(i)
      val newpi = (piAvg * 0.01) + (oldpi * 0.09)
      nn_p += newpi
    }
    nn_p.toArray
  }

  /**
   * NNbp是后向传播
   * 计算权重的平均偏导数
   */
  def NNbp(
    train_nnff: RDD[(NNLabel, Array[BDM[Double]])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]],
    bc_nn_p: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Array[BDM[Double]] = {
    // 第n层偏导数：d(n)=-(y-a(n))*f'(z)，sigmoid函数f'(z)表达式:f'(z)=f(z)*[1-f(z)]
    // sigm: d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
    // {'softmax','linear'}: d{n} = - nn.e;
    val train_data5 = train_nnff.map { f =>
      val nn_a = f._1.nna
      val error = f._1.error
      val dn = ArrayBuffer[BDM[Double]]()
      val nndn = bc_config.value.output_function match {
        case "sigm" =>
          val fz = nn_a(bc_config.value.layer - 1)
          (error * (-1.0)) :* (fz :* (1.0 - fz))
        case "linear" =>
          error * (-1.0)
      }
      dn += nndn
      (f._1, f._2, dn)
    }
    // 第n-1至第2层导数：d(n)=-(w(n)*d(n+1))*f'(z) 
    val train_data6 = train_data5.map { f =>
      // 假设 f(z) 是sigmoid函数 f(z)=1/[1+e^(-z)]，f'(z)表达式，f'(z)=f(z)*[1-f(z)]    
      // 假设 f(z) tanh f(z)=1.7159*tanh(2/3.*A) ，f'(z)表达式，f'(z)=1.7159 * 2/3 * (1 - 1/(1.7159)^2 * f(z).^2)   
      //val di = ArrayBuffer( BDM((1.765226346140333)))
      //      val nn_a = ArrayBuffer[BDM[Double]]()
      //      val a1=BDM((1.0,0.312605257000000,0.848582961000000,0.999014768000000,0.278330771000000,0.462701179000000))
      //      val a2= BDM((1.0,0.838091550300577,0.996782915917104,0.118033012437165))
      //      val a3= BDM((2.18788852054974))
      //      nn_a += a1
      //      nn_a += a2
      //      nn_a += a3
      val nn_a = f._1.nna
      val di = f._3
      val dropout = f._2
      for (i <- (bc_config.value.layer - 2) to 1 by -1) {
        // f'(z)表达式
        val nnd_act = bc_config.value.activation_function match {
          case "sigm" =>
            val d_act = nn_a(i) :* (1.0 - nn_a(i))
            d_act
          case "tanh_opt" =>
            val fz2 = (1.0 - ((nn_a(i) :* nn_a(i)) * (1.0 / (1.7159 * 1.7159))))
            val d_act = fz2 * (1.7159 * (2.0 / 3.0))
            d_act
        }
        // 稀疏度惩罚误差计算:-(t/p)+(1-t)/(1-p)
        // sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        val sparsityError = if (bc_config.value.nonSparsityPenalty > 0) {
          val nn_pi1 = bc_nn_p.value(i)
          val nn_pi2 = (bc_config.value.sparsityTarget / nn_pi1) * (-1.0) + (1.0 - bc_config.value.sparsityTarget) / (1.0 - nn_pi1)
          val Bm1 = new BDM(nn_pi2.rows, 1, Array.fill(nn_pi2.rows * 1)(1.0))
          val sparsity = BDM.horzcat(Bm1, nn_pi2 * bc_config.value.nonSparsityPenalty)
          sparsity
        } else {
          val nn_pi1 = bc_nn_p.value(i)
          val sparsity = BDM.zeros[Double](nn_pi1.rows, nn_pi1.cols + 1)
          sparsity
        }
        // 导数：d(n)=-( w(n)*d(n+1)+ sparsityError )*f'(z) 
        // d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act;
        val W1 = bc_nn_W.value(i)
        val nndi1 = if (i + 1 == bc_config.value.layer - 1) {
          //in this case in d{n} there is not the bias term to be removed  
          val di1 = di(bc_config.value.layer - 2 - i)
          val di2 = (di1 * W1 + sparsityError) :* nnd_act
          di2
        } else {
          // in this case in d{i} the bias term has to be removed
          val di1 = di(bc_config.value.layer - 2 - i)(::, 1 to -1)
          val di2 = (di1 * W1 + sparsityError) :* nnd_act
          di2
        }
        // dropoutFraction
        val nndi2 = if (bc_config.value.dropoutFraction > 0) {
          val dropouti1 = dropout(i)
          val Bm1 = new BDM(nndi1.rows: Int, 1: Int, Array.fill(nndi1.rows * 1)(1.0))
          val dropouti2 = BDM.horzcat(Bm1, dropouti1)
          nndi1 :* dropouti2
        } else nndi1
        di += nndi2
      }
      di += BDM.zeros(1, 1)
      // 计算最终需要的偏导数值：dw(n)=(1/m)∑d(n+1)*a(n)
      //  nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
      val dw = ArrayBuffer[BDM[Double]]()
      for (i <- 0 to bc_config.value.layer - 2) {
        val nndW = if (i + 1 == bc_config.value.layer - 1) {
          (di(bc_config.value.layer - 2 - i).t) * nn_a(i)
        } else {
          (di(bc_config.value.layer - 2 - i)(::, 1 to -1)).t * nn_a(i)
        }
        dw += nndW
      }
      (f._1, di, dw)
    }
    val train_data7 = train_data6.map(f => f._3)

    // Sample a subset (fraction miniBatchFraction) of the total data
    // compute and sum up the subgradients on this subset (this is one map-reduce)
    val initgrad = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      val init1 = if (i + 1 == bc_config.value.layer - 1) {
        BDM.zeros[Double](bc_config.value.size(i + 1), bc_config.value.size(i) + 1)
      } else {
        BDM.zeros[Double](bc_config.value.size(i + 1), bc_config.value.size(i) + 1)
      }
      initgrad += init1
    }
    val (gradientSum, miniBatchSize) = train_data7.treeAggregate((initgrad, 0L))(
      seqOp = (c, v) => {
        // c: (grad, count), v: (grad)
        val grad1 = c._1
        val grad2 = v
        val sumgrad = ArrayBuffer[BDM[Double]]()
        for (i <- 0 to bc_config.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (grad, count)
        val grad1 = c1._1
        val grad2 = c2._1
        val sumgrad = ArrayBuffer[BDM[Double]]()
        for (i <- 0 to bc_config.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c1._2 + c2._2)
      })
    // 求平均值
    val gradientAvg = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      val Bm1 = gradientSum(i)
      val Bmavg = Bm1 :/ miniBatchSize.toDouble
      gradientAvg += Bmavg
    }
    gradientAvg.toArray
  }

  /**
   * NNapplygrads是权重更新
   * 权重更新
   */
  def NNapplygrads(
    train_nnbp: Array[BDM[Double]],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]],
    bc_nn_vW: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Array[Array[BDM[Double]]] = {
    // nn = nnapplygrads(nn) returns an neural network structure with updated
    // weights and biases
    // 更新权重参数：w=w-α*[dw + λw]    
    val W_a = ArrayBuffer[BDM[Double]]()
    val vW_a = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      val nndwi = if (bc_config.value.weightPenaltyL2 > 0) {
        val dwi = train_nnbp(i)
        val zeros = BDM.zeros[Double](dwi.rows, 1)
        val l2 = BDM.horzcat(zeros, dwi(::, 1 to -1))
        val dwi2 = dwi + (l2 * bc_config.value.weightPenaltyL2)
        dwi2
      } else {
        val dwi = train_nnbp(i)
        dwi
      }
      val nndwi2 = nndwi :* bc_config.value.learningRate
      val nndwi3 = if (bc_config.value.momentum > 0) {
        val vwi = bc_nn_vW.value(i)
        val dw3 = nndwi2 + (vwi * bc_config.value.momentum)
        dw3
      } else {
        nndwi2
      }
      // nn.W{i} = nn.W{i} - dW;
      W_a += (bc_nn_W.value(i) - nndwi3)
      // nn.vW{i} = nn.momentum*nn.vW{i} + dW;
      val nnvwi1 = if (bc_config.value.momentum > 0) {
        val vwi = bc_nn_vW.value(i)
        val vw3 = nndwi2 + (vwi * bc_config.value.momentum)
        vw3
      } else {
        bc_nn_vW.value(i)
      }
      vW_a += nnvwi1
    }
    Array(W_a.toArray, vW_a.toArray)
  }

  /**
   * nneval是进行前向传播并计算输出误差
   * 计算神经网络中的每个节点的输出值，并计算平均误差;
   */
  def NNeval(
    batch_xy: RDD[(BDM[Double], BDM[Double])],
    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Double = {
    // NNff是进行前向传播
    // nn = nnff(nn, batch_x, batch_y);
    val train_nnff = NeuralNet.NNff(batch_xy, bc_config, bc_nn_W)
    // error and loss
    // 输出误差计算
    val loss1 = train_nnff.map(f => f._1.error)
    val (loss2, counte) = loss1.treeAggregate((0.0, 0L))(
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
    val Loss = loss2 / counte.toDouble
    Loss * 0.5
  }
}

