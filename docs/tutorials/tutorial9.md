# Shape completion using Gaussian process regression

In this tutorial we will show how GP regression can be used to predict a missing parts of a shape.

##### Preparation

As in the previous tutorials, we start by importing some commonly used objects and initializing the system.

```scala
import scalismo.geometry._
import scalismo.common._
import scalismo.ui.api._
import scalismo.mesh._
import scalismo.io.{StatisticalModelIO, MeshIO, LandmarkIO}
import scalismo.statisticalmodel._
import scalismo.numerics.UniformMeshSampler3D
import scalismo.kernels._
import breeze.linalg.{DenseMatrix, DenseVector}

scalismo.initialize()
implicit val rng = scalismo.utils.Random(42)

val ui = ScalismoUI()
```

We also load a dataset that we want to reconstruct. In this case, it is a face without nose:

```scala
val noseless = MeshIO.readMesh(new java.io.File("datasets/noseless.stl")).get

val targetGroup = ui.createGroup("target")
ui.show(targetGroup, noseless,"noseless")
```

Finally, we also load the face model.

```scala
val smallModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/model.h5")).get 
```

## Enlarging the flexibility of a shape model

The model, which we just loaded, was built from only a small dataset. Therefore, the chances that it manages to
reconstruct the missing nose properly are rather slim.

To increase the shape variability of the model, we add smooth some additional smooth shape deformations,
modelled by a GP with symmetric Gaussian kernel. The code should be familiar from the previous tutorials.

```scala
val scalarValuedKernel = GaussianKernel[_3D](30) * 10.0

case class XmirroredKernel(kernel : PDKernel[_3D]) extends PDKernel[_3D] {
  override def domain = RealSpace[_3D]
  override def k(x: Point[_3D], y: Point[_3D]) = kernel(Point(x(0) * -1f ,x(1), x(2)), y)
}

def symmetrizeKernel(kernel : PDKernel[_3D]) : MatrixValuedPDKernel[_3D] = {
   val xmirrored = XmirroredKernel(kernel)
   val k1 = DiagonalKernel(kernel, 3)
   val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)  
   k1 + k2
}

val gp = GaussianProcess[_3D, EuclideanVector[_3D]](symmetrizeKernel(scalarValuedKernel))
val lowrankGP = LowRankGaussianProcess.approximateGP(gp, UniformMeshSampler3D(smallModel.referenceMesh, 200), numBasisFunctions = 30)
val model = StatisticalMeshModel.augmentModel(smallModel, lowrankGP)

val modelGroup = ui.createGroup("face model")
val ssmView = ui.show(modelGroup, model, "model")
```

The new model should now contain much more flexibility, while still preserving the typical face-specific deformations.

*Note: This step here is mainly motivated by the fact that we only have 10 face examples available to build the model. However,
even if sufficient data is available, it might still be a good idea to slighly enlarge the flexibility of a model
before attempting a reconstruction of missing parts. If just gives the model some extra slack to explain shape variations, which
have not been prominent in the dataset*.

Equipped with our new model, we will now use our model to perform the reconstruction. We proceed in two steps:

1. We will fit the face model to the given partial face using Gaussian process regression.
   This means that we seek to find a model instance that resembles well enough the given partial shape.
2. We will then take the nose part of the fit as a reconstruction proposal for the missing nose.

As we saw previously, to perform GP regression we need observations of the deformation vectors at some points.
How we can obtain such vectorsrs automatically, will be discussed in [Tutorial 10](./tutorial10.html). Here,   
we have done this already in a separate step and saved 200 corresponding points as landmarks, which we will now load and visualize:

```scala
val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/modelLandmarks.json")).get
val referencePoints : Seq[Point[_3D]] = referenceLandmarks.map(lm => lm.point)
val referenceLandmarkViews = referenceLandmarks.map(lm => ui.show(modelGroup, lm, s"lm-${lm.id}"))


val noselessLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/noselessLandmarks.json")).get
val noselessPoints : Seq[Point[_3D]] = noselessLandmarks.map(lm => lm.point)
val noselessLandmarkViews = noselessLandmarks.map(lm => ui.show(targetGroup, lm, s"lm-${lm.id}"))
```

By indicating these correspondences above, we just indicated how each selected point of the
model should be deformed to its corresponding point on the target mesh.
In other words, we **observed** a few deformation vectors at the selected model points.

```scala
val domain = UnstructuredPointsDomain(referencePoints.toIndexedSeq)
// domain: UnstructuredPointsDomain[_3D] = scalismo.common.UnstructuredPointsDomain3D@9644dc34
val deformations = (0 until referencePoints.size).map(i => noselessPoints(i) - referencePoints(i) )
// deformations: collection.immutable.IndexedSeq[EuclideanVector[_3D]] = Vector(
//   EuclideanVector3D(
//     -6.808479309082031,
//     -5.5187530517578125,
//     -1.5841598510742188
//   ),
//   EuclideanVector3D(-4.5506591796875, 0.017253875732421875, 1.9810104370117188),
//   EuclideanVector3D(-5.884559631347656, 3.4085350036621094, -6.472417831420898),
//   EuclideanVector3D(
//     -4.289102554321289,
//     -0.25388336181640625,
//     6.2558746337890625
//   ),
//   EuclideanVector3D(-3.9964599609375, 3.270242691040039, 2.15728759765625),
//   EuclideanVector3D(3.537139892578125, 6.198089599609375, 5.472877502441406),
//   EuclideanVector3D(1.7101821899414062, -11.340240478515625, -1.62274169921875),
//   EuclideanVector3D(4.536685943603516, -1.5926475524902344, 0.2605438232421875),
//   EuclideanVector3D(-0.29347991943359375, -11.498748779296875, -6.166748046875),
//   EuclideanVector3D(0.9505023956298828, -11.77044677734375, 3.342548370361328),
//   EuclideanVector3D(-3.770599365234375, 6.8009033203125, 3.5061264038085938),
//   EuclideanVector3D(-3.510894775390625, -2.260957717895508, -7.199741363525391),
//   EuclideanVector3D(0.6191596984863281, 1.7126121520996094, 7.8465423583984375),
//   EuclideanVector3D(-6.029518127441406, 7.852027893066406, -6.743669509887695),
//   EuclideanVector3D(4.450469970703125, -5.679145812988281, -6.782912254333496),
//   EuclideanVector3D(-0.718144416809082, 1.191279411315918, 7.491905212402344),
//   EuclideanVector3D(
//     -3.1832046508789062,
//     -1.5409507751464844,
//     -0.8160171508789062
//   ),
//   EuclideanVector3D(2.1937408447265625, 1.5413446426391602, 4.707771301269531),
//   EuclideanVector3D(0.7241592407226562, 2.7885398864746094, 7.8438262939453125),
//   EuclideanVector3D(-6.486053466796875, -1.9876556396484375, 7.102943420410156),
//   EuclideanVector3D(6.573799133300781, 3.7882652282714844, 1.6320838928222656),
//   EuclideanVector3D(
//     5.517463684082031,
//     -0.15950965881347656,
//     -5.8751068115234375
//   ),
//   EuclideanVector3D(-0.9969406127929688, -3.6084823608398438, 7.487541198730469),
//   EuclideanVector3D(
//     -5.6077728271484375,
//     -2.3572616577148438,
//     -4.603706359863281
//   ),
//   EuclideanVector3D(-5.1880950927734375, 3.550199508666992, 1.3362083435058594),
//   EuclideanVector3D(0.39674949645996094, 6.5591278076171875, 3.8452606201171875),
// ...
val defField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](domain, deformations)
// defField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = <function1>
ui.show(modelGroup, defField, "partial_Field")
// res2: ShowInScene.ShowInSceneDiscreteFieldOfVectors.View = VectorFieldView(
//   partial_Field
// )
```

We can now perform GP regression and retrieve the rest of the deformations fitting our observations:

```scala
val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.5)

val regressionData = for ((refPoint, noselessPoint) <- referencePoints zip noselessPoints) yield {
    val refPointId = model.referenceMesh.pointSet.findClosestPoint(refPoint).id 
    (refPointId, noselessPoint, littleNoise) 
}

val posterior = model.posterior(regressionData.toIndexedSeq)

val posteriorGroup = ui.createGroup("posterior-model")
ui.show(posteriorGroup, posterior, "posterior")
```

With this posterior model, we get a normal distribution of faces satisfying our observations by having the selected characteristic points at the indicated positions.


Finally, as we are interested in the nose region only, we marginalize our posterior to obtain a posterior nose model as we did in a previous tutorial:

```scala

val nosePtIDs = model.referenceMesh.pointSet.pointIds.filter { id =>
  (model.referenceMesh.pointSet.point(id) - model.referenceMesh.pointSet.point(PointId(8152))).norm <= 42
}

val posteriorNoseModel = posterior.marginal(nosePtIDs.toIndexedSeq)

val posteriorNoseGroup = ui.createGroup("posterior-nose-model")
ui.show(posteriorNoseGroup, posteriorNoseModel, "posteriorNoseModel")
```

