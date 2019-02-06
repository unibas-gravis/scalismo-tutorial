# Shape completion using Gaussian process regression

In this tutorial we will show how GP regression can be used to predict a missing parts of a shape.

##### Preparation

As in the previous tutorials, we start by importing some commonly used objects and initializing the system.

```scala
import scalismo.geometry._
import scalismo.common._
import scalismo.ui.api._
import scalismo.mesh._
import scalismo.io.{StatismoIO, MeshIO, LandmarkIO}
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
val smallModel = StatismoIO.readStatismoMeshModel(new java.io.File("datasets/model.h5")).get 
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

val gp = GaussianProcess[_3D, Vector[_3D]](symmetrizeKernel(scalarValuedKernel))
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
// deformations: collection.immutable.IndexedSeq[Vector[_3D]] = Vector(
//   Vector3D(-6.808479309082031, -5.5187530517578125, -1.5841598510742188),
//   Vector3D(-4.5506591796875, 0.017253875732421875, 1.9810104370117188),
//   Vector3D(-5.884559631347656, 3.4085350036621094, -6.472417831420898),
//   Vector3D(-4.289102554321289, -0.25388336181640625, 6.2558746337890625),
//   Vector3D(-3.9964599609375, 3.270242691040039, 2.15728759765625),
//   Vector3D(3.537139892578125, 6.198089599609375, 5.472877502441406),
//   Vector3D(1.7101821899414062, -11.340240478515625, -1.62274169921875),
//   Vector3D(4.536685943603516, -1.5926475524902344, 0.2605438232421875),
//   Vector3D(-0.29347991943359375, -11.498748779296875, -6.166748046875),
//   Vector3D(0.9505023956298828, -11.77044677734375, 3.342548370361328),
//   Vector3D(-3.770599365234375, 6.8009033203125, 3.5061264038085938),
//   Vector3D(-3.510894775390625, -2.260957717895508, -7.199741363525391),
//   Vector3D(0.6191596984863281, 1.7126121520996094, 7.8465423583984375),
//   Vector3D(-6.029518127441406, 7.852027893066406, -6.743669509887695),
//   Vector3D(4.450469970703125, -5.679145812988281, -6.782912254333496),
//   Vector3D(-0.718144416809082, 1.191279411315918, 7.491905212402344),
//   Vector3D(-3.1832046508789062, -1.5409507751464844, -0.8160171508789062),
//   Vector3D(2.1937408447265625, 1.5413446426391602, 4.707771301269531),
//   Vector3D(0.7241592407226562, 2.7885398864746094, 7.8438262939453125),
//   Vector3D(-6.486053466796875, -1.9876556396484375, 7.102943420410156),
//   Vector3D(6.573799133300781, 3.7882652282714844, 1.6320838928222656),
//   Vector3D(5.517463684082031, -0.15950965881347656, -5.8751068115234375),
//   Vector3D(-0.9969406127929688, -3.6084823608398438, 7.487541198730469),
//   Vector3D(-5.6077728271484375, -2.3572616577148438, -4.603706359863281),
//   Vector3D(-5.1880950927734375, 3.550199508666992, 1.3362083435058594),
//   Vector3D(0.39674949645996094, 6.5591278076171875, 3.8452606201171875),
//   Vector3D(8.216743469238281, 0.47994422912597656, -2.099668502807617),
//   Vector3D(8.2755126953125, -2.141468048095703, -3.583367347717285),
//   Vector3D(1.0731258392333984, 0.9502897262573242, 3.9115676879882812),
//   Vector3D(-0.08221435546875, -10.040321350097656, 3.07525634765625),
//   Vector3D(-5.159210205078125, -2.007059097290039, -5.148160934448242),
//   Vector3D(-6.98736572265625, 2.603485107421875, -1.1500740051269531),
//   Vector3D(2.934390068054199, -12.068061828613281, 9.071632385253906),
//   Vector3D(6.189750671386719, 3.8783187866210938, -7.682294845581055),
//   Vector3D(-0.8288421630859375, 6.7830963134765625, 0.6582717895507812),
//   Vector3D(-0.8912258148193359, 5.501190185546875, 4.0577545166015625),
//   Vector3D(3.5532455444335938, -6.203619003295898, -6.658527374267578),
//   Vector3D(-6.30718994140625, 2.540966033935547, -4.019872665405273),
//   Vector3D(-6.524726867675781, -1.87506103515625, 9.850906372070312),
//   Vector3D(-5.01519775390625, 1.1915521621704102, -0.2905921936035156),
//   Vector3D(5.279701232910156, 5.6873016357421875, 3.5121726989746094),
//   Vector3D(-6.5226287841796875, -0.6888041496276855, -3.679011344909668),
//   Vector3D(-6.541086196899414, -2.041461944580078, 11.670982360839844),
//   Vector3D(2.2548675537109375, -13.320632934570312, -4.434486389160156),
//   Vector3D(-5.588233947753906, 2.8534488677978516, -2.7899856567382812),
//   Vector3D(3.39300537109375, -13.782302856445312, -1.6893424987792969),
//   Vector3D(1.6024303436279297, -1.5331153869628906, 7.151496887207031),
//   Vector3D(-3.777383804321289, -2.2366180419921875, 7.047172546386719),
// ...
val defField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], Vector[_3D]](domain, deformations)
// defField: DiscreteField[_3D, UnstructuredPointsDomain[_3D], Vector[_3D]] = <function1>
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

