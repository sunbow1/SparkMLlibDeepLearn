package CNN

import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV
}
import org.apache.spark.rdd.RDD

/**
 * label：目标矩阵
 * features：特征矩阵
 * predict_label：预测矩阵
 * error：误差
 */
case class PredictCNNLabel(label: BDM[Double], features: BDM[Double], predict_label: BDM[Double], error: BDM[Double]) extends Serializable

class CNNModel(
  val cnn_layers: Array[CNNLayers],
  val cnn_ffW: BDM[Double],
  val cnn_ffb: BDM[Double]) extends Serializable {

  /**
   * 返回预测结果
   *  返回格式：(label, feature,  predict_label, error)
   */
  def predict(dataMatrix: RDD[(BDM[Double], BDM[Double])]): RDD[PredictCNNLabel] = {
    val sc = dataMatrix.sparkContext
    val bc_cnn_layers = sc.broadcast(cnn_layers)
    val bc_cnn_ffW = sc.broadcast(cnn_ffW)
    val bc_cnn_ffb = sc.broadcast(cnn_ffb)
    // CNNff是进行前向传播
    val train_cnnff = CNN.CNNff(dataMatrix, bc_cnn_layers, bc_cnn_ffb, bc_cnn_ffW)
    val rdd_predict = train_cnnff.map { f =>
      val label = f._1
      val nna1 = f._2(0)(0)
      val nnan = f._4
      val error = f._4 - f._1
      PredictCNNLabel(label, nna1, nnan, error)
    }
    rdd_predict
  }

  /**
   * 计算输出误差
   * 平均误差;
   */
  def Loss(predict: RDD[PredictCNNLabel]): Double = {
    val predict1 = predict.map(f => f.error)
    // error and loss
    // 输出误差计算
    val loss1 = predict1
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
    val Loss = (loss2 / counte.toDouble) * 0.5
    Loss
  }

}