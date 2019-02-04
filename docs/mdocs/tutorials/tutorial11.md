```tut:invisible

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
```

# Model fitting with Iterative Closest Points

The goal in this tutorial is to learn how to fit a shape model to a target surface using Iterative Closest Points (ICP) and obtain correspondences by doing so.

In case you would like to learn more about the syntax of the Scala programming language, have a look at the **Scala Cheat Sheet**. This document is accessible under: 

*Documents -> Scala Cheat Sheet*

###### Note: for optimal performance and to keep the memory footprint low, we recommend to restart Scalismo Lab at the beginning of every tutorial.


### Meaning of model fitting

Let's load a mesh to fit:
```tut:silent
val target = MeshIO.readMesh(new File("datasets/target.stl")).get
show(target,"target")
```
and load our model:

```tut:silent
val model = StatismoIO.readStatismoMeshModel(new File("datasets/bfm.h5")).get
show(model, "model")
```

As you can see in the 3D scene, we are currently displaying an instance of our model (the mean), that does not resemble our target face. The goal in shape model fitting is therefore to find an instance of our shape model that resembles at best the given target face.

By doing so, we obtain correspondences between the points of our model and the fit to the target mesh. With such correspondences, many applications become possible such as the reconstruction we performed in a previous tutorial, shape segmentation, anomaly detection and many others.

### Iterative Closest Points (ICP) and GP regression

In a previous tutorial, we saw how to use ICP in combination with Procrustes analysis in order to find the best rigid transform aligning one mesh to another. In that case the method proceeded as follows: 

1. Suggest **candidate** correspondences between the mesh to be aligned and the target one, by attributing the closest point on the target mesh as a candidate.

2. Solve for the best rigid transform between the moving mesh and the target mesh using Procrustes analysis.

3. Transform the moving mesh using the retrieved transform and loop to step 1 if the result is not aligned with the target (or if we didn't reach the limit number of iterations)

In that case we noticed that, with every iteration, the quality of the suggested candidate correspondences gradually increased, leading to better and better rigid transformations.


Here, we will perform exactly the same steps to fit our model to the target while substituting Procrustes analysis with Gaussian process regression.

#### Fitting a few characteristic points 

To keep things simple, let us first try to find correspondences for a few characteristic points: 

```tut:silent
val landmarks = LandmarkIO.readLandmarksJson[_3D](new File("datasets/icpLandmarks.json")).get
addLandmarksTo(landmarks, "model")
```

Here we loaded a set of easily identifiable points on the model (see in 3D scene), for which we will seek corresponding points on the target mesh.

**Obtaining candidate correspondences**

Let's start by defining our candidate correspondences attribution method and use it on the loaded points:

```tut:silent
def attributeCorrespondences(pts : Seq[Point[_3D]]) : Seq[Point[_3D]] = {
  pts.map{pt => target.findClosestPoint(pt).point}
}

val candidates = attributeCorrespondences(landmarks.map { l => l.point }) 
show(candidates, "candidates")
```

Notice here that the defined function is very similar to the one we defined for the rigid alignment ICP introduced in a previous tutorial. Given a set of points, in this case belonging to an instance of our shape model, we attribute the **closest point** on the target as a candidate correspondence to each one of these points. The returned sequence of points therefore contains the candidate correspondences to the input points *pts*.

When visualizing the attributed candidate correspondences, we see that we get an unsatisfying initialization, where some points such as some corners of the eyes are rather well initialized, while other points such as the tip of the nose are still poorly fit.


The first main idea behind ICP is, even though the candidate correspondences are still not perfect, to nevertheless proceed to step 2 and use them in a **GP regression** to find the best model instance explaining the observed deformations.


Let us start by visualizing the partial deformation field (or candidate observations) that we will feed to our regression:


```tut:silent
val pointIds = landmarks.map{l => model.mean.findClosestPoint(l.point).id}.toIndexedSeq
val modelPts = pointIds.map(id => model.referenceMesh.point(id) )
val domain = UnstructuredPointsDomain[_3D](modelPts.toIndexedSeq)
val values =  (modelPts.zip(candidates)).map{case (mPt, pPt) => pPt -mPt}
val field = DiscreteVectorField(domain, values.toIndexedSeq)  
show(field, "deformations")
```

Let's now use this candidate partial deformation for regression: 
```tut:silent
val littleNoise = NDimensionalNormalDistribution(Vector(0,0,0), SquareMatrix((1f,0,0), (0,1f,0), (0,0,1f)))

def fitModel(pointIds: IndexedSeq[PointId],candidateCorresp: Seq[Point[_3D]]) :TriangleMesh = { 
  val trainingData = (pointIds zip candidateCorresp).map{ case (mId, pPt) => 
    (mId, pPt, littleNoise)
  }
  val posterior = model.posterior(trainingData.toIndexedSeq)
  posterior.mean
}

val fit = fitModel(pointIds, candidates)
show(fit, "fit")
```

Here we started by defining our noise that we will use for the regression.

We then defined the *fitModel* function that, when given a sequence of identifiers of model points and their candidate correspondence positions, computes a GP regression based on the resulting deformation field and returns the model instance fitting at best the candidate deformations.


As we are now limiting our interest to the few characteristic points, let us visualize their new position on the obtained fit: 


```tut:silent
val fittedPoints = pointIds.map(id => fit.point(id))
show(fittedPoints, "fittedPoints")
remove("fit"); remove("candidates"); remove("deformations")
```
(make the target slightly transparent and color *fittedPoints* to visualize better)

Notice here that the set of points we just displayed are **real corresponding** points to the set of chosen landmarks on the model as they are taken from the model fit using the same point identifiers.

When comparing the positions of these points with regard to the target mesh, we see that they are still far from their corresponding position on the target (corner of eye, tip of nose, etc ..). 

However, when considering the initial position of these points, that are the loaded landmarks positions, one might argue that we are at a better position than where we started.


This is where the second important idea of ICP comes again into play: **iteration**.
If we were now to repeat the same operations above (attribute candidate correspondences and use them to obtain a model fit), this time however starting from the new points positions (*newPoints* in the scene), we could hope to get an even better model fit.

```tut:silent
def recursion(currentPoints : Seq[Point[_3D]], nbIterations : Int) : Unit= {

  val candidates = attributeCorrespondences(currentPoints)
  val fit = fitModel(pointIds, candidates)  

  val newPoints= pointIds.map(id => fit.point(id))
  remove("newPoints")
  show(newPoints, "newPoints")

  if(nbIterations> 0) {
    Thread.sleep(3000)
    recursion(newPoints, nbIterations - 1)
  }
}
```

Here we defined a recursive function that repeats the exact same procedure we performed above for a given number of iterations. At each iteration, candidate correspondences are attributed to the input list of point positions. A Gaussian process regression is then performed and a new position for the points of interest is computed. 

The new point positions are then used in the recursive function call to indicate the new starting position.

At every iteration, the new positions of the corresponding characteristic points are displayed (with a 3 second delay for us to be able to follow the fitting progress).


When now calling the recursive function for 5 iterations, starting from the loaded landmark positions, we should see the characteristic points gradually getting to better positions.
(Make sure to look at the point positions after executing this)

```tut:silent
recursion( pointIds.map(id => model.mean.point(id)), 5) 
```

The final position of the characteristic points is then considered to be our model fit. These points are therefore the obtained correspondences with the target shape.

##### Question: the found corresponding points are not always on the target surface, why is that?

###### Answer: it quite often happens that the model does not exactly fit every point of the target shape, therefore resulting in some model points not lying on the target. The reason for this could be that the model fit did not converge (not enough iterations or local minimum), or that the model is not flexible enough to explain the target. In any case, since we are searching for corresponding points *x + u(x)* where *x* is a reference point and *u* is a deformation from our model, we still consider *x + u(x)* as the corresponding point. If we were to take the closest point on the target to *x + u(x)* as a corresponding point, we would obtain correspondences that are not explained by our model, which is generally not desired.


##### Exercise: call the recursive function again while this time increasing the number of iterations. Does that impact the quality of the fitting in this case? If not, why not? 

###### Answer: similar to the case in the rigid alignment ICP, the ICP fitting here also converges to a local minimum (be it a good or a bad one), out of which it does not exit. Increasing the number of iterations therefore does not have any effect on the quality of the result.

#### ICP with more points

Let us now do the exact same operations above, this time with much more points of interest.


```tut:silent
val pointSamples = UniformMeshSampler3D(model.mean, 5000, 42).sample.map(s => s._1)
val pointIds = pointSamples.map{s => model.mean.findClosestPoint(s).id}
val initialPositions = pointIds.map(id => model.mean.point(id))
show(initialPositions, "more_points")
```

Here we start by uniformly sampling 5000 points on our model and retrieving the identifiers of the closest mesh vertices to these points. These vertices will now be the points for which we will seek correspondences.

```tut:silent
def recursion(currentPoints : Seq[Point[_3D]], nbIterations : Int) : Unit = {
  println("iterations left " + nbIterations)
  val candidates = attributeCorrespondences(currentPoints)
  val fit = fitModel(pointIds, candidates)  
  remove("fit")
  show(fit,"fit")

  val newPoints= pointIds.map(id => fit.point(id))

  if(nbIterations> 0) {
    Thread.sleep(2000)
    recursion(newPoints, nbIterations - 1)
  }
}
```

We then define a recursive fitting function again, very similar to the one above, with the only difference that we are now visualizing the entire fitted mesh at every iteration and no longer limiting our attention to the fitted points.

We can now call the recursion with 10 iterations to obtain our **model fit**; that is an instance of our face model that resembles the given target mesh.

```tut:silent
remove("more_points")
recursion(initialPositions , 10) // color the target to visualize better
```

##### Exercise: given the obtained fit, retrieve the corresponding points to all the points of the face model and display the resulting deformation field. Hint: this would require modifying the return value of the recursive function.

```tut:book
def recursion(currentPoints : Seq[Point[_3D]], nbIterations : Int) : Iterator[Point[_3D]]= {

  val candidates = attributeCorrespondences(currentPoints)
  val fit = fitModel(pointIds, candidates)  

  val newPoints= pointIds.map(id => fit.point(id))

  if(nbIterations> 0) {
    Thread.sleep(1000)
    recursion(newPoints, nbIterations - 1)
  } else {
     fit.points 
  }
}

val fittedPoints = recursion(initialPositions, 10)
val pts = fittedPoints.toIndexedSeq
val dom = UnstructuredPointsDomain(pts)

val defs = (pts zip model.referenceMesh.points.toIndexedSeq).map(t =>  t._1 - t._2)

show( DiscreteVectorField(dom, defs), "defs")
```
<br><br>

```tut:invisible
gui.close()
```