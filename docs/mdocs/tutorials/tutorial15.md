{% include head.html %}

# Bayesian Model Fitting - The basic framework

In this tutorial we show how the MCMC framework, which was introduced in the previous
tutorial can be used for shape model fitting. 

We will illustrate it by computing a posterior of a shape model, 
given a set of corresponding landmark pairs. This is the same setup, as we have 
discussed in the tutorial about Gaussian process regression. The difference is, 
that here we will also allow for rotation and translation of the model. In this setting, 
it is not possible anymore to compute the posterior analytically. Rather, approximation methods, such as
using Markov-chain monte carlo methods are our only hope.  

In this tutorial we show not only a working example, but also how to make it 
computationally efficient. Making the individual parts as efficient as possible is 
important in sampling approaches, as we need to produce many samples to get accurate 
estimates. 

##### Preparation

As in the previous tutorials, we start by importing some commonly used objects and 
initializing the system. 

```scala mdoc:silent
import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry._
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RigidTransformationSpace, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Memoize

implicit val rng = scalismo.utils.Random(42)
scalismo.initialize()

val ui = ScalismoUI()
```

### Loading and visualizing the data

In a first step, we load and visualize all the data that we need. 
This is first the statistical model:

```scala mdoc:silent
  val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/bfm.h5")).get

  val modelGroup = ui.createGroup("model")
  val modelView = ui.show(modelGroup, model, "model")
```

In this example, we will fit the model such that a set of landmark points, defined on the model, coincide
with a set of landmark points defined on a target face. We load and visualize the corresponding landmark data:
  
```scala mdoc:silent
  val modelLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/modelLM_mcmc.json")).get
  val modelLmViews = ui.show(modelGroup, modelLms, "modelLandmarks")
  modelLmViews.foreach(lmView => lmView.color = java.awt.Color.BLUE)

  val targetGroup = ui.createGroup("target")
  
  val targetLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/targetLM_mcmc.json")).get
  val targetLmViews = ui.show(targetGroup, targetLms, "targetLandmarks")
  modelLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)
```

In the following we will refer to the points on the model using their point id, while the target points
are represented as physical points.   
```scala mdoc:silent
  val modelLmIds =  modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
  val targetPoints = targetLms.map(l => l.point)
```

The set of correspondences are then given by
```scala mdoc:silent
    val landmarkNoiseVariance = 4.0
    val uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )
  val correspondences = modelLmIds.zip(targetPoints).map(modelIdWithTargetPoint => {
    val (modelId, targetPoint) =  modelIdWithTargetPoint
    (modelId, targetPoint, uncertainty)
  })
    
```


### The parameter class

```scala mdoc:silent
  case class PoseAndShapeParameters(translationParameters: EuclideanVector[_3D],
                                    rotationParameters: (Double, Double, Double),
                                    modelCoefficients: DenseVector[Double],
                                    private val rotationCenter: Point[_3D]
                                   )  {
    def poseTransformation : RigidTransformation[_3D] = {

      val translation = TranslationTransform(translationParameters)
      val rotation = RotationTransform(
        rotationParameters._1,
        rotationParameters._2,
        rotationParameters._3,
        rotationCenter
      )
      RigidTransformation(translation, rotation)
    }
    def fullTransformation = {
      model.instance(modelCoefficients).transform(poseTransformation)
    }
  }

```

### Evaluators: Modelling the target density

```scala mdoc:silent
  case class ShapePriorEvaluator(model: StatisticalMeshModel)
    extends DistributionEvaluator[PoseAndShapeParameters] {
    override def logValue(theta: PoseAndShapeParameters): Double = {
      model.gp.logpdf(theta.modelCoefficients)
    }
  }
```

The simple version 
``` scala mdoc:silent
  case class CorrespondenceEvaluatorSimple(model: StatisticalMeshModel,
                                          correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
      extends DistributionEvaluator[PoseAndShapeParameters] {
  
      override def logValue(theta: PoseAndShapeParameters): Double = {
  
        val currModelInstance = model.instance(theta.modelCoefficients).transform(theta.poseTransformation)
  
        val likelihoods = correspondences.map( correspondence => {
          val (id, targetPoint, uncertainty) = correspondence
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint
  
          uncertainty.logpdf(observedDeformation.toBreezeVector)
        })
  
  
        val loglikelihood = likelihoods.sum
        loglikelihood
      }
    }
```

Here is the more efficient version. 1: use a var to compute the likelihood, 2. marginalize the model

```scala mdoc:silent
case class CorrespondenceEvaluator(model: StatisticalMeshModel,
                                     correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
    extends DistributionEvaluator[PoseAndShapeParameters] {
    val (modelIds, _, _) = correspondences.unzip3

    val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
    val newCorrespondences = correspondences.map(idWithTargetPoint => {
      val (id, targetPoint, uncertainty) = idWithTargetPoint
      val modelPoint = model.referenceMesh.pointSet.point(id)
      val newId = marginalizedModel.referenceMesh.pointSet.findClosestPoint(modelPoint).id
      (newId, targetPoint, uncertainty)
    })


    override def logValue(theta: PoseAndShapeParameters): Double = {

      val currModelInstance = marginalizedModel.instance(theta.modelCoefficients).transform(theta.poseTransformation)

      val likelihoods = newCorrespondences.map( correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        val modelInstancePoint = currModelInstance.pointSet.point(id)
        val observedDeformation = targetPoint - modelInstancePoint

        uncertainty.logpdf(observedDeformation.toBreezeVector)
      })


      val loglikelihood = likelihoods.sum
      loglikelihood
    }
  }
```
  
We also introduce a caching class, which ...
``` scala mdoc:silent
  implicit class CachedEvaluator(evaluator: DistributionEvaluator[PoseAndShapeParameters]) {
      val memoizedLogValue = Memoize(evaluator.logValue, 10)
  
      def cached() = {
        new DistributionEvaluator[PoseAndShapeParameters] {
          override def logValue(sample: PoseAndShapeParameters): Double = {
            memoizedLogValue(sample)
          }
        }
      }
    }
```

The evaluators can now be constructed as follows
```scala mdoc:silent
val likelihoodEvaluator = CorrespondenceEvaluator(model, correspondences).cached()
val priorEvaluator = ShapePriorEvaluator(model).cached()

val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)
```

### The proposal generator

```scala mdoc:silent
case class ShapeUpdateProposal(paramVectorSize : Int, stdev: Double)
    extends ProposalGenerator[PoseAndShapeParameters]  with TransitionProbability[PoseAndShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Double](paramVectorSize) * stdev
    )


    override def propose(theta: PoseAndShapeParameters): PoseAndShapeParameters = {
      val perturbation = perturbationDistr.sample()
      val thetaPrime = theta.copy(modelCoefficients = theta.modelCoefficients + perturbationDistr.sample)
      thetaPrime
    }

    override def logTransitionProbability(from: PoseAndShapeParameters, to: PoseAndShapeParameters) = {
      val residual = to.modelCoefficients - from.modelCoefficients
      perturbationDistr.logpdf(residual)
    }

  }
```

```scala mdoc:silent
  case class RotationUpdateProposal(stdev: Double) extends
    ProposalGenerator[PoseAndShapeParameters]  with TransitionProbability[PoseAndShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * stdev)

    def propose(theta: PoseAndShapeParameters): PoseAndShapeParameters= {
      val perturbation = perturbationDistr.sample
      val newRotationParameters = (
        theta.rotationParameters._1 + perturbation(0),
        theta.rotationParameters._2 + perturbation(1),
        theta.rotationParameters._3 + perturbation(2)
      )
      theta.copy(rotationParameters = newRotationParameters)
    }

    override def logTransitionProbability(from: PoseAndShapeParameters, to: PoseAndShapeParameters) = {
      val residual = DenseVector(
        to.rotationParameters._1 - from.rotationParameters._1,
        to.rotationParameters._2 - from.rotationParameters._2,
        to.rotationParameters._3 - from.rotationParameters._3
      )
      perturbationDistr.logpdf(residual)
    }
  }
```

```scala mdoc:silent
  case class TranslationUpdateProposal(stdev: Double) extends
    ProposalGenerator[PoseAndShapeParameters]  with TransitionProbability[PoseAndShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution( DenseVector.zeros(3),
      DenseMatrix.eye[Double](3) * stdev)

    def propose(theta: PoseAndShapeParameters): PoseAndShapeParameters= {
      val newTranslationParameters = theta.translationParameters + EuclideanVector.fromBreezeVector(perturbationDistr.sample())
      theta.copy(translationParameters = newTranslationParameters)
    }

    override def logTransitionProbability(from: PoseAndShapeParameters, to: PoseAndShapeParameters) = {
      val residual = to.translationParameters - from.translationParameters
      perturbationDistr.logpdf(residual.toBreezeVector)
    }
  }
  
  ```
  
The final proposal
 ```scala mdoc:silent
val shapeUpdateProposal = ShapeUpdateProposal(model.rank, 0.1)
val rotationUpdateProposal = RotationUpdateProposal(0.01)
val translationUpdateProposal = TranslationUpdateProposal(1.0)
val generator = MixtureProposal.fromProposalsWithTransition(
 (0.6, shapeUpdateProposal),
 (0.2, rotationUpdateProposal),
 (0.2, translationUpdateProposal)
)
   
 ```
 
 
#### Building the Markov Chain


```scala mdoc:silent
  class Logger extends AcceptRejectLogger[PoseAndShapeParameters] {
    var numAccepted = 0
    var numRejected = 0

    override def accept(current: PoseAndShapeParameters, sample: PoseAndShapeParameters, generator: ProposalGenerator[PoseAndShapeParameters], evaluator: DistributionEvaluator[PoseAndShapeParameters]): Unit = {
      numAccepted += 1
    }
    override def reject(current: PoseAndShapeParameters, sample: PoseAndShapeParameters, generator: ProposalGenerator[PoseAndShapeParameters], evaluator: DistributionEvaluator[PoseAndShapeParameters]): Unit = {
      numRejected += 1
    }
    def acceptanceRatio() = {
      numAccepted / (numAccepted + numRejected).toDouble
    }
  }
```

```scala mdoc:silent
  val chain = MetropolisHastings(generator, posteriorEvaluator)


  def computeCenterOfMass(mesh : TriangleMesh[_3D]) : Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
  }

    val initialParameters = PoseAndShapeParameters(
    EuclideanVector(0, 0, 0),
    (0.0, 0.0, 0.0),
    DenseVector.zeros[Double](model.rank),
      computeCenterOfMass(model.mean)
    )
    
  val mhIterator = chain.iterator(initialParameters, new Logger())
```

```scala mdoc:silent
   val samplingIterator = for((theta, itNum) <- mhIterator.zipWithIndex) yield {
      println("iteration " + itNum)
      if (itNum % 500 == 0) {
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = theta.modelCoefficients
        modelView.shapeModelTransformationView.poseTransformationView.transformation = theta.poseTransformation
      }
      theta
    }
    
  val thetas = samplingIterator.drop(1000).take(10000).toIndexedSeq
  ```
  
  
### Analysing the results

We will now compare the variance at the points with the one we would have gotton using only the posterior
  
  ```scala mdoc:silent
  
   def computeMean(id: PointId): Point[_3D] = {
      var mean = EuclideanVector(0, 0, 0)
      for (theta <- thetas) yield {
        mean += model.instance(theta.modelCoefficients).pointSet.point(id).toVector
      }
      (mean * 1.0 / thetas.size).toPoint
    }
  
    def computeCovarianceFromSamples(id: PointId, mean: Point[_3D]): SquareMatrix[_3D] = {
      var cov = SquareMatrix.zeros[_3D]
      for (theta <- thetas) yield {
        val instance = model.instance(theta.modelCoefficients)
        val v = instance.pointSet.point(id) - mean
        cov += v.outer(v)
      }
      cov * (1.0 / thetas.size)
    }
  
    println("before computing posterior")
    val realPosterior = model.posterior(correspondences.toIndexedSeq)
    // compute uncertainty at landmark positions
    println("here ")
    for ((id, _, _) <- correspondences) yield {
      val covShapeAndPose = computeCovarianceFromSamples(id, computeMean(id))
      println(s"posterior variance computed  for id (shape and pose) $id  = ${covShapeAndPose(0,0)}, ${covShapeAndPose(1,1)}, ${covShapeAndPose(2,2)}")
      val covAnalyticShapeOnly = realPosterior.cov(id, id)
      println(s"posterior variance computed by analytic posterior (shape only) for id $id = ${covAnalyticShapeOnly(0,0)}, ${covAnalyticShapeOnly(1,1)}, ${covAnalyticShapeOnly(2,2)}")
    }
```

Furthermore, we will compute the variance in translation that we get
```scala mdoc

```

```scala mdoc:invisible
ui.close()
```
