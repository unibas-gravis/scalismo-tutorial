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
    import scalismo.common.ScalarArray.implicits._
    scalismo.initialize()

    import ch.unibas.cs.gravis.shapemodelling.SimpleAPIWrapperForTut._
```

# Finding correspondence in an image

The goal in this tutorial is to understand how to use intensity models to find candidate correspondences between a shape model and an image.

In case you would like to learn more about the syntax of the Scala programming language, have a look at the **Scala Cheat Sheet**. This document is accessible under: 

*Documents -> Scala Cheat Sheet*

###### Note: for optimal performance and to keep the memory footprint low, we recommend to restart Scalismo Lab at the beginning of every tutorial.


## Correspondence between shapes and images

Let's start by loading the image and the model to fit:

```tut:silent
val image = ImageIO.read3DScalarImage[Short](new File("datasets/PaolaMRI.vtk")).get
show(image, "image")

val model = StatismoIO.readStatismoMeshModel(new File("datasets/asmData/model.h5")).get
show(model, "model")
```

##### Exercise: go to a slice view (*View -> Perspective -> Y slice*) and zoom in on the model contour (that is the intersection of the model mesh with the slice). How well is the face model fitting to the face of the patient in the MRI?

As you could hopefully see for yourself, the current model instance, that is the mean, does not follow well the contour of the face depicted in the image and therefore does not **fit** to the image.

Similar to fitting a shape model to a target mesh, the goal of model fitting to an image is to find an instance of the model that resembles the face in the image. 

The problem is now however more complicated as we do not have a geometrical representation of the target shape. Instead, the target face shape is indicated in the **intensities** of the image (that is itself defined on a regular grid of points). 

Hence, the challenge when fitting our model to the image will then be to identify, based on the image intensities, where the points of the model correspond on the image.


##### Exercise: slice through the MRI image and try to locate the tip of the nose. Click a landmark at the position of the tip of the nose in the image, and a landmark on the tip of the nose on the model mean.

###### Note: in case clicking landmarks did not work for you, you can add the landmarks programmatically by executing the code below.

```tut:silent
//execute this only if you were unable to click landmarks
addLandmarksTo(Seq(Landmark("A", model.mean.point(PointId(8277)))), "model")
addLandmarksTo(Seq(Landmark("A",  Point(114.97f,37.07f,246.09f))), "image")
```
Now that you indicated the corresponding point position between the tip of the nose in the model and in the image, we can perform GP regression and retrieve a model instance fitting the indicated correspondence.

```tut:silent
val tipModel = getLandmarksOf("model").get.head.point
val tipImage = getLandmarksOf("image").get.head.point
val tipNoseID = model.mean.findClosestPoint(tipModel).id

val littleNoise = NDimensionalNormalDistribution(Vector(0,0,0), SquareMatrix((1f,0,0), (0,1f,0), (0,0,1f)))
val trainingData = IndexedSeq((tipNoseID, tipImage, littleNoise))
val posterior = model.posterior( trainingData )
show(posterior, "posterior")
```
###### Note: in case the code above resulted in an Exception "UnsupportedOperationException: empty.head", please make sure to click the landmarks requested in the exercise above.

##### Exercise: check the contour of the displayed posterior model with regard to the image. Any progress in the quality of the fit? How about in the nose region?

Naturally, one should not expect much from a fit based on a single point. However, you hopefully get the idea that if we could perform the same operation for many more points, we would eventually get a decent fit.

For the rest of the tutorial, we will try to perform such a correspondence attribution **automatically**.

### Fitting a single point: the tip of the nose

**Where are the candidate positions for the tip of the nose?**

Given that we have a statistical shape model, we can model the potential positions of the tip of the nose by building a marginal model for this point: 

```tut:silent
val tipMarginal = model.marginal(IndexedSeq(tipNoseID))
val candidatePoints = (0 to 100).map(i => tipMarginal.sample.point(PointId(0)))
show(candidatePoints, "candidates")
```
Here we sampled 100 instances out of our tip marginal and displayed 100 different positions that the tip of the nose could assume.  Therefore, when searching for *candidate* positions for the tip of the nose in the image, it is sensible to search for **the best** position among these sampled points.

##### Exercise: display the plotted tip positions together with the image (in the 3D view, use the Z slider to center the slice on the candidate points). Are all points valid candidates to be a tip of a nose on the image?

We will now try to discriminate between the different candidate points and evaluate, based on the image intensities, if each candidate could be valid tip of a nose or not.

For this, we need an **intensity model**.

#### Intensity model

When given a point of interest on the shape, an intensity model indicates the intensity values normally associated with this point in sample images.

##### Exercise: evaluate the intensity at the tip of the nose in the MRI image. You can do this by maintaining the Ctrl key pressed and hovering over the image. The intensity value will then be displayed in the bottom left corner of the viewer window.

Clearly, to build such an intensity model, we need a sample of MRI images where we can locate the tip of the nose.


Below we load a dataset of face scans in correspondence:

```tut:silent
val faces = new File("datasets/asmData/faces/").listFiles.sortBy(_.getName).map{ f => MeshIO.readMesh(f).get }
show(faces.head ,"face1") // display the first face

remove("image"); remove("posterior") // clean up the scene
remove("candidates"); remove("model")
```

and a set of MRI images **fitting** to the loaded faces: 

```tut:silent
val mri = new File("datasets/asmData/mri/").listFiles.sortBy(_.getName).map{ 
  f => ImageIO.read3DScalarImage[Short](f).get
}

show(mri.head, "mri1") // display the first image
```

##### Exercise: go to a slice view and zoom in on the contour of the first face of our dataset. How well is it following the face contour in the corresponding MRI image?

As you could hopefully verify for yourself, the contour of the first mesh follows nicely the contour of the face depicted in the intensities of the MRI image (except maybe for the ear region which is irrelevant in our case).

##### Exercise: verify that the face meshes are fitting the MRI images for the rest of the dataset.

```tut:book
show(mri(1), "mri_2")
show(faces(1), "face_2")
// verify visually and repeat for others indexes: 2,3
remove("mri_2"); remove("face_2")
```

Given that the loaded faces are in correspondence, we can locate the tip of the nose in all 4 faces. Below, we locate it on the first face mesh and add it as a landmark:

```tut:silent
 val tipFace1 = faces.head.point(tipNoseID)
 addLandmarksTo(Seq(Landmark("tip",tipFace1)), "face1")
```
##### Exercise: zoom in on the added landmark (you can do this by selecting the landmark then right clicking *-> Center slices here*) and evaluate the intensity at the tip in the first MRI. How does it compare to our target value?

###### Answer: depending on the point you selected to evaluate the intensity of the nose tip on the target image, the value should be around 100 and 140. In this particular case, we notice that the intensity value at the landmark is 84, as it lies slightly outside of the face contour. This is due to the fact that the corresponding face is not **perfectly** matching the image contour.


Now that we have the 3D point position of the tip of the nose, we can evaluate its intensity in the corresponding MRI programmatically:

```tut:silent
val continuousImage1 = mri.head.interpolate(3)
val tipIntensity = continuousImage1(tipFace1)
```
Notice here that we started by interpolating the image first. This is due to the fact that the tip of the nose might not necessarily be one of the grid points on which the image is defined.

Given that we have a dataset of fitting faces and MRIs, we can do the the same for other tips of noses in the dataset and collect their different intensities:

```tut:silent
val continuousImages : IndexedSeq[ScalarImage[_3D]] = mri.map{ 
  im => im.interpolate(3)
}

val faceAndMri = faces zip continuousImages

val tipIntensities = faceAndMri.map{ case (face, mri) => 
  val tipPosition : Point[_3D] = face.point(tipNoseID)
  val intensityVal : Float = mri(tipPosition)
  DenseVector(intensityVal) 
}
```

Now *tipIntensities* contains 4 intensity values observed at tips of the nose in 4 different MRI images.

##### Exercise: if we were to model the intensities of the different nose tips using a normal distribution, what would be the mean and standard deviation of the distribution?
###### Answer: the answer is just below :)

Now that we have our different sample intensity values, we can compute our intensity model by approximating the samples with a normal distribution using a method of the *MultivariateNormalDistribution* object in Scalismo:


```tut:silent
val intensityModel = MultivariateNormalDistribution.estimateFromData(tipIntensities)

println("mean tip intensity " + intensityModel.mean)
println("standard deviation tip intensity " + math.sqrt(intensityModel.cov(0,0)))
```
##### Question: the standard deviation seems to be rather high. Why do you think this is the case?

###### Answer: this is due to the fact that the face surfaces are not always **perfectly** matching the corresponding MRI images as we noticed above. In some cases the nose tip is slightly outside/inside of the image contour therefore resulting in lower/higher intensity values.


####  Evaluating the fitness of candidate positions

Let us now go back to the goal of evaluating the fitness of candidate tip positions using to the intensity model.

To do so, we evaluate the image intensity at every candidate point position and compute its [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) to the intensity model:

```tut:silent
val continuousImage = image.interpolate(3)

val mahalDistances = candidatePoints.map { p => 
  val candidateIntensity = DenseVector(continuousImage(p))
  intensityModel.mahalanobisDistance(candidateIntensity) 
}
```
Here again, we started by interpolating our target MRI image as the candidate points do not belong to the discrete grid on which the MRI image is defined.

The *mahalDistances* variable now contains the distance of each candidate point to the intensity model.

The smaller the distance value, the more fit is the candidate to be a tip of a nose according to the intensity model.


Let us now visualize these Mahalanobis distances on the candidate points:  

```tut:silent
val domain = UnstructuredPointsDomain(candidatePoints)
val field = DiscreteScalarField[_3D, Double](domain, mahalDistances.toArray)
show(image, "image")
show(field, "mahal")
remove("face1") ; remove("mri1") // cleanup
```

Here we started by creating a discrete scalar field, that is a 3D point cloud with a scalar value associated to each point. To do so, we specified the domain over which the field is defined, that are the candidate points. We also specify the values associated with every point of the field, which are the corresponding Mahalanobis distances.

##### Exercise: visualize the scalar field in the scene. The more a point's color goes to red, the lower is its Mahalanobis distance to the intensity model. Where are the good points lying?

###### Answer: in the paragraph below

## Intensity models with neighborhood

As you could hopefully see for yourself, the rather simplistic intensity model we computed above leads to many *false positives*. These are candidate nose tips considered to be good by the intensity model, while clearly lying too  far inside of the head.

To narrow down our search to more relevant candidates, we need to extend our intensity model to take **the neighborhood** of the candidate points into account.

The notion of neighborhood can of course take many forms: a squared patch around each point, an elliptic patch, etc ... We will choose in the following to adopt the same neighborhood notion as the Active Shape Model algorithm and consider a neighborhood of points along the vector normal to our shape at the point of interest:

```tut:silent
val normal1 : Vector[_3D] = faces.head.normalAtPoint(tipFace1)
val neighbors = (-4 to 4).map(i => tipFace1 + normal1 * i)
show(neighbors, "normalNeighbors")
show(mri.head, "mri1")
remove("image");remove("mahal") // cleanup
```

###### Note: Switch to the 3D view to visualize the neighbor points

Here we started by retrieving the normal vector to our first face at the tip of the nose. We then regularly sampled and displayed 9 points along this normal, each spaced with 1mm (as our vector is normalized).

Let's now define a function to extract the image intensities at these neighbor points:

```tut:silent
 def getFeatureVector(image: ScalarImage[_3D], tip: Point[_3D], normalDirection : Vector[_3D]) : DenseVector[Float] = {
  val neighbors = (-4 to 4).map { i => tip + normalDirection * i }
  val intensities = neighbors.map(p => image(p))
  DenseVector(intensities.toArray)
}
```

Here, we defined a function that when given a continuous image (the MRI), a point position in the image and the direction of the normal vector to the corresponding face at the tip of the nose, returns a vector containing 
the intensity values at the neighborhood of the given point.

We can now use this function to extract such a vector for every tip of nose in our dataset and build an intensity model, not only based on the single intensity at the tip point, but on the intensities in a neighborhood region of the tip of the nose:

```tut:silent
 val tipFeatures = faceAndMri.map { case (face, mri) =>
  val tip = face.point(tipNoseID)
  val normal = face.normalAtPoint(tip)
  getFeatureVector(mri, tip, normal)
}
val betterIntensityModel = MultivariateNormalDistribution.estimateFromData(tipFeatures)
val mean = betterIntensityModel.mean
```
##### Exercise: observe the mean intensity vector. What is the general tendency of the intensities at the neighborhood of the tip of the nose?

###### Answer: when looking at the mean vector DenseVector(212.0841, 255.41866, 224.4656, 173.90057, 117.96514, 95.278206, 49.601017, 30.171265, 23.168406) one sees the progression of the intensity values from the inside of the head towards the outside values with the first value being an intensity measured at a neighbor of the tip inside the head and the last value at a neighbor outside. Such a profile captures much better the intensity information around the nose tip than the (rather noisy) single intensity at the tip point.


Notice that, to evaluate the Mahalanobis distance of our candidate points, we also need to extract such an intensity vector for each candidate:

```tut:silent
val normalOnMean = model.mean.normalAtPoint(tipModel)

val newMahalDistances = candidatePoints.map { candidate => 
  val candidateIntensityVec = getFeatureVector(continuousImage, candidate, normalOnMean)
  betterIntensityModel.mahalanobisDistance(candidateIntensityVec) 
}

val newField = DiscreteScalarField[_3D, Double](domain, newMahalDistances.toArray)
show(newField, "new mahal")
show(image, "image")
remove("mri1"); remove("normalNeighbors") // cleanup

```

In this case, we choose the normal direction to be that of the normal to the tip of the nose on the mean mesh.

##### Exercise: zoom in on the new displayed scalar field. Where are most of the red points located now?

###### Answer: most of the red points are now located at *border* points between the inside and the outside of the face as all these points match well with the learned intensity profiles with the first half of the values being high values and the second half rather low ones.

To finish our correspondence search, let us take the candidate point with the minimal Mahalanobis distance as our corresponding tip of the nose in the MRI:
  
```tut:silent
val tipInMRI = candidatePoints.minBy { candidate =>
  val candidateIntensityVec = getFeatureVector(continuousImage, candidate, normalOnMean)
  betterIntensityModel.mahalanobisDistance(candidateIntensityVec)
}
remove("new mahal")
show(Seq(tipInMRI), "BestCandidate") 
```

Notice here how we simply replaced the Scala collection's *map* method call by a *minBy* call (also a method of Collections) to now simply retrieve the element in the candidate list with the minimal Mahalanobis distance.

As you can see, using this intensity model including the neighborhood information, we could now **automatically** locate the tip of the nose in the target MRI.

<br><br>

```tut:invisible
gui.close
```