``tut:invisible

import scalismo.io.MeshIO
import scalismo.io.ImageIO
import scalismo.io.StatismoIO
import scalismo.io.LandmarkIO
import java.io.File
import scalismo.geometry._
import scalismo.mesh._
import scalismo.registration._
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import scala.math._
import scalismo.geometry.Landmark

import scalismo.image.filter._
import scalismo.common._
import scalismo.kernels._
import scalismo.numerics._
import scalismo.registration._
import scalismo.common._
import scalismo.statisticalmodel._
import scalismo.kernels._
import scalismo.statisticalmodel.dataset._
import scalismo.statisticalmodel.asm._
import scalismo.statisticalmodel.asm.ActiveShapeModel._
import scalismo.image._
import scalismo.ui.api.SimpleAPI._
  import scalismo.ui.api.Show

scalismo.initialize()

import ch.unibas.cs.gravis.shapemodelling.SimpleAPIWrapperForTut._

# Shape completion using Gaussian process regression

The goal in this tutorial is to learn how to use GP regression to predict missing parts of a shape.

In case you would like to learn more about the syntax of the Scala programming language, have a look at the **Scala Cheat Sheet**. This document is accessible under:

*Documents -> Scala Cheat Sheet*

###### Note: for optimal performance and to keep the memory footprint low, we recommend to restart Scalismo Lab at the beginning of every tutorial.


## Enlarging the flexibility of a shape model

Let's start by loading an incomplete face that we need to reconstruct:

```tut:silent
val noseless = MeshIO.readMesh(new File("datasets/noseless.stl")).get
show(noseless,"noseless")

As you can see, the nose is missing. In the remainder of the tutorial, we will use a simple face model built from 10 face scans to reconstruct the missing nose:

```tut:silent val littleModel = StatismoIO.readStatismoMeshModel(new File("datasets/model.h5")).get show(littleModel, "littleModel")

As this model was built from a very little dataset, chances that it manages to reconstruct the missing nose properly are rather slim.

To increase the shape variability of the model, we can combine it with a symmetric Gaussian kernel:

```tut:silent
val zeroMean = VectorField(RealSpace[_3D], (pt:Point[_3D]) => Vector(0,0,0))
val scalarValuedKernel = GaussianKernel[_3D](30) * 10

case class XmirroredKernel(ker : PDKernel[_3D]) extends PDKernel[_3D] {
  override def domain = RealSpace[_3D]
  override def k(x: Point[_3D], y: Point[_3D]) = ker(Point(x(0) * -1f ,x(1), x(2)), y)
}

def SymmetrizeKernel(ker : PDKernel[_3D]) : MatrixValuedPDKernel[_3D,_3D] = {
   val xmirrored = XmirroredKernel(ker)
   val k1 = DiagonalKernel(ker)
   val k2 = DiagonalKernel(xmirrored * -1f, xmirrored, xmirrored)  
   k1 + k2
}

val sim = SymmetrizeKernel(scalarValuedKernel)

val gp = GaussianProcess(zeroMean, sim)

val model = StatisticalMeshModel.augmentModel(littleModel, gp, 50)
show(model, "model")

Here, we started by creating a Gaussian process yielding symmetric smooth deformations as we did in a previous tutorial. We then used this GP to enlarge the flexibility of our little face model by calling the augmentModel method of the StatisticalMeshModel object where we indicated the shape model to enlarge, the Gaussian process we just built, and the desired number of eigenfunctions to be kept.

The augment method used above is a utility function that combines two steps we have seen in previous tutorials:

    It starts by building a new continuous Gaussian process where the covariance function is now the sum of two kernels. In our case, these are the covariances learned from data and the kernel function of the indicated GP (the symmetric Gaussian GP in our case).

    It then performs a low-rank decomposition of the combined GP (by means of a KL-expansion) and retains only the indicated number of eigenfunctions (50 in our case).

As a result, we obtain a more flexible shape model containing variations from both sample data and a generic smooth symmetric shape model.
Exercise: compare the old and new model by sampling random faces from both.
Model fitting when given correspondence

We will now use our model to perform the reconstruction as follows:

    We will try to fit the face model to the given partial face using Gaussian process regression. This means that we seek to find a model instance that resembles well enough the given partial shape.

    We will then take the nose part of the fit as a reconstruction proposal for the missing nose.

As we saw previously, to perform GP regression we need observations of the deformation vectors at some points. We can obtain such observations by indicating some correspondences manually:

```tut:silent remove("littleModel") val lms = LandmarkIO.readLandmarksJson_3D.get addLandmarksTo(lms, "noseless")

##### Exercise: click landmarks on the model mean that correspond to the landmarks we just added to our *noseless* mesh. Attention:  it is \*very\* important that the order of the landmarks in the two lists is the same. Therefore, make sure to click the model landmarks according to the alphabetical order in the figure below:


######Note: in case you were unable to click the landmarks, you can add them programmatically by executing the code below
```tut:silent
// execute this only if you were unable to click the landmarks
val lm = LandmarkIO.readLandmarksJson[_3D](new File("datasets/noseFittingModelLms.json")).get
addLandmarksTo(lm,"model")

```tut:silent val modelPts : Seq[Point[_3D]] = getLandmarksOf("model").get.map{lm => lm.point} val noselessPts = getLandmarksOf("noseless").get.map{lm => lm.point}

By indicating these correspondences above, we just indicated how each selected point of the model should be deformed to its corresponding point on the target mesh. In other words, we **observed** a few deformation vectors at the selected model points.

##### Exercise: visualize the partial deformation field that we obtain by clicking these correspondences.

```tut:book
val modelPtIds : Seq[PointId] = modelPts.map(p => model.mean.findClosestPoint(p).id )
val referencePoints = modelPtIds.map(id => model.referenceMesh.point(id)) // deformations are defined on the reference points

val domain = UnstructuredPointsDomain(referencePoints.toIndexedSeq)
val deformations = (0 until referencePoints.size).map(i => noselessPts(i) - referencePoints(i) )
val defField = DiscreteVectorField(domain, deformations)
show(defField, "partial_Field")

We can now perform GP regression and retrieve the rest of the deformations fitting our observations:

```tut:silent val littleNoise = NDimensionalNormalDistribution(Vector(0,0,0), SquareMatrix((0.5f,0,0), (0,0.5f,0), (0,0,0.5f)))

val trainingData = (modelPts zip noselessPts).map{ case (mPt, nPt) => (model.mean.findClosestPoint(mPt).id, nPt, littleNoise) }

val posterior = model.posterior(trainingData.toIndexedSeq) show(posterior, "posterior")

In this case, we performed the regression directly at the *StatisticalMeshModel* level and retrieved the posterior mesh model fitting our observations.

With this posterior model, we get a normal distribution of faces satisfying our observations by having the selected characteristic points at the indicated positions.

##### Exercise: sample a few random faces from the posterior model and compare the position of the selected landmarks points to the corresponding points on the target mesh.  How well do the sampled faces resemble our target?


### What if we had more correspondences?

As you could hopefully see for yourself, even with the little amount of indicated correspondences, we start to obtain face model instances that resemble our target mesh. To obtain more similar face instances, we need more correspondences:

```tut:silent
val modelLMs = LandmarkIO.readLandmarksJson[_3D](new File("datasets/modelLandmarks.json")).get
addLandmarksTo( modelLMs, "model")

val noselessLMs = LandmarkIO.readLandmarksJson[_3D](new File("datasets/noselessLandmarks.json")).get
addLandmarksTo( noselessLMs, "noseless")

Here, we just loaded 200 pre-indicated corresponding landmarks from file and added them to both the model and the target mesh.

We can now reiterate our posterior model computation, this time however, with more landmarks:

```tut:silent val modelLandmarks = getLandmarksOf("model").get val noselessLandmarks = getLandmarksOf("noseless").get

val trainingData = (modelLandmarks zip noselessLandmarks).map{ case (mLm, nLm) => (model.mean.findClosestPoint(mLm.point).id, nLm.point, littleNoise) }

val betterPosterior = model.posterior(trainingData.toIndexedSeq)

show(betterPosterior, "betterPosterior")

##### Exercise: sample a few random faces from the new posterior model. How much variability is left in the model? In which region are the sampled shapes varying the most? (Hint: you can check the first principal components of the shape model to visualize the biggest variations)
###### Answer: hopefully you managed to see that most of the remaining variability is now at the nose region. This is very noticeable when varying the first principal component of the posterior model.

Finally, as we are interested in the nose region only, let us marginalize our posterior to obtain a posterior nose model as we did in a previous tutorial:

```tut:silent
val nosePtIDs = model.referenceMesh.pointIds.filter { id =>
  (model.referenceMesh.point(id) - model.referenceMesh.point(PointId(8152))).norm <= 42
}

val posteriorNoseModel = betterPosterior.marginal(nosePtIDs.toIndexedSeq)
show(posteriorNoseModel, "posteriorNoseModel")

Exercise: hide all the objects in the 3D scene, except for the noseless face and the nose model. Sample a few instances from the model by changing the coefficients of the principal components and also by drawing random samples. How do the reconstructions look?
Exercise: the borders of the computed nose model instances do not always perfectly match the incomplete mesh. How would you amend this problem?
Answer: one possible solution could be to click more landmarks on the edges that need to be stitched together.

As you could see, using more corresponding points leads to a better nose reconstruction. The question however remains: How did we obtain the 200 correspondences we used for the second posterior model?
Note: to perform the reconstruction in this tutorial, we used a small face model (10 faces) that we augmented with a symmetric Gaussian kernel. Much better reconstruction results can however be obtained when using a statistical shape model built from a larger dataset. Such a model, the Basel Face Model, is freely available for download, for non-commercial purposes.

<br><br>

tut:invisible gui.close