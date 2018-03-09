
# Quickstart:

In this document, we will guide you through the task of creating shape models using Gaussian processes,
and to fit this model to a mesh using Scalismo's registration framework.

In the following, we assume that you successfully [setup](https://github.com/unibas-gravis/scalismo/wiki/Setup-a-project-using-Scalismo) a project making use of Scalismo.

In case you started your own sbt project from scratch (and did not start from the [tutorial project](https://github.com/unibas-gravis/scalismo-tutorial)), 
please download and extract the example data [here](http://shapemodelling.cs.unibas.ch/data/face-data.zip).

## Imports

There are some basic imports, which we will need in most scalismo applications. 
The other packagges will be imported directly when needed. 

```scala mdoc
import scalismo.common._
import scalismo.geometry._
import breeze.linalg.DenseVector
```


## Initializing the system
At the beginning of every program that is using Scalismo, you will need to do two things:

1. Initialize the system and load all the system's native libraries that are used by scalismo
2. Choose a source of randomness for Scalismo. This can either be done using a default source or by defining an implicit 
value with a (seeded) random source. The latter is especially useful when reproducible results should be obtained, 
such as when you are debugging your program.  

The following code shows how Scalismo can be initialized: 
```scala mdoc

scalismo.initialize()
import scalismo.utils.Random.implicits._ // this uses the default random source 
//implicit val rng = Random(seed = 1024L) // used to provide a seeded source of randomness     
```


## Loading and visualizing a mesh

We start by loading and visualizing the reference mesh, which we will later use as the
domain for our Gaussian Process model. Visualizing is done using ```scalismo-ui```:
```scala mdoc
import scalismo.ui.api._
```

We also need the following imports

```scala mdoc
import java.io.File
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh 
import java.awt.Color
```

First we load a mesh from a file. 
```scala mdoc
val referenceMesh = MeshIO.readMesh(new File("datasets/quickstart/facemesh.stl")).get
```
The get function at the end of the line is there because readMesh could fail.
By calling get, we indicate that we are not interested in handling possible errors and just want retrieve the result. If the result is not available, an exception is thrown.
(The function .get should only be used in research code, where error handling is 
not of big importance. In more serious software project, proper error handling should be done.)

Scalismo was designed to make it simple to visualize objects. Visualization is provided by
```ScalismoUI```. To start ScalismoUI, we write
```scala mdoc
val ui = ScalismoUI()
```

We also create a group in the ui. A group groups together different views of the same object.
```scala mdoc
val modelGroup = ui.createGroup("model")
```
 
To visualize the mesh that we just loaded we call
```scala mdoc
val refMeshView = ui.show(modelGroup, referenceMesh, "referenceMesh")
```

This adds the surface mesh to the created group "modelGroup" under the name "referenceMesh" and visualizes
the mesh. Every call to show return a "view" object, which lets us control the visualization
properties of the object. This object allows us, for example to change its color
```scala mdoc
refMeshView.color = Color.RED
```

or its opacity

```scala mdoc
refMeshView.opacity = 0.5
```

*You can also change the visualization properties of this object by selecting the object in the scene tree of the gui. To remove or make an object invisible, right click on an object in the scene tree.*

To remove the mesh again from the scene we call
```scala mdoc
refMeshView.remove
```

## Building a Gaussian process shape model

Now we are ready to build a Gaussian Process model.
We create a new Gaussian process that models 3D vector fields. 
Let's start, however, with the necessary imports

```scala mdoc
import scalismo.kernels._
import scalismo.statisticalmodel._
import scalismo.numerics._
```
We define the mean of the Gaussian process to be the zero vector field in 3D.
As a covariance function we use a Gaussian kernel and choose to treat the x,y,z component
of the vector field to be uncorrelated. This last property is achieved by using a ```DiagonalKernel```.

```scala mdoc
val mean = VectorField(RealSpace[_3D], (_ : Point[_3D]) => Vector.zeros[_3D])
val kernel = DiagonalKernel[_3D](GaussianKernel(sigma = 40) * 50.0, outputDim = 3)
val gp = GaussianProcess(mean, kernel)
```

Before we can use the Gaussian process for modelling shapes, we need to perform a low-rank
approximation to obtain a parametric, tractable definition. Here, we approximate
the first 50 basis functions. The approximation uses a finite sample of
points on the mesh to approximate the basis function. Which points are used is determined
by the sampler we provide. Here we choose to sample arbitrary points of the mesh.

```scala mdoc
val meshSampler = UniformMeshSampler3D(
    referenceMesh, 
    numberOfPoints = 300)

val lowRankGP = LowRankGaussianProcess.approximateGP(
    gp, 
    meshSampler, 
    numBasisFunctions = 50)
```

We can sample random function (i.e. vector fields) from the resulting
randomFun by calling

```scala mdoc
val randomFun = lowRankGP.sample
```

randomFun is a proper function, which we can evaluate on any point in 3D.
For example, we can call

```scala mdoc
val aRandomValue = randomFun(Point(7.0, 15.0, 1.0))
```

to evaluate it at the point (7, 15, 1). The resulting Gaussian process can now be used to define a shape model. This is 
 achieved by applying it to all the points of the reference mesh as follows:
 
```scala mdoc
val shapeModel = StatisticalMeshModel(referenceMesh, lowRankGP)
```
As we see here, a shape model is simply the combination of a referenceMesh with a Gaussian proces.
This is also used in ScalismoUi to visualize Gaussian process models. This view is also reflected
in the ui. When we show a model using
```scala mdoc
val ssmView = ui.show(modelGroup, shapeModel, "SSM")
```
we see that in the scene tree the reference mesh and a GP object are added.
To visualize the shape variations, we can now manipulate the coefficients of the
gp. In the ui, click on the gp object that is now
shown in the scene tree. You can sample random functions by clicking the random button,
or explore the shape space more efficiently by changing the sliders.



## Registration
We can now use this model to perform a registration of a target mesh. We start by importing the necessary classes and 
by loading the target mesh and display it.

```scala mdoc
import scalismo.registration._
```

```scala mdoc
val targetGroup = ui.createGroup("target")
val targetMesh = MeshIO.readMesh(new File("datasets/quickstart/face-2.stl")).get
val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")
```

*To visualize a registration, it is best to change the perspective in the graphical user interface to "orthogonal slices". You can find this functionality in the "View -> Perspective" menu.*

To define a registration, we need to define four things:
1. a `transformation space` that models the possible transformations of the reference surface (or the ambient space)
2. a `metric` to measure the distance between the model an the target surface. 
3. a `regularizer`, which penalizes unlikely transformations
4. an `optimizer` 

For non-rigid registration we usually model the possible transformations using a Gaussian process. We use the Gaussian process that
we have defined above to define the trasnformation space.  
```scala mdoc
 val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
```

As a metric, we use a simple mean squares metric. Currently, all metrics that are available in scalismo are implemented as 
image to image metrics. These can, however, easily be used for surface registration by representing the surface as  a distance image. 
In addition to the images, the metric also needs to know the possible transformations (as modelled by the transformation space) and 
a sampler. The sampler determines the points where the metric is evaluated. In our case we choose uniformely sampled points on the
reference mesh. 
```scala mdoc
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = UniformMeshSampler3D(referenceMesh, numberOfPoints = 1000)
    val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
```

As an optimizer, we choose an LBFGS Optimizer
```scala mdoc
val optimizer = LBFGSOptimizer(maxNumberOfIterations = 100)
```
and for regularization we choose to penalize the L2 norm using the `L2Regularizer`:
```scala mdoc
val regularizer = L2Regularizer(transformationSpace)
``` 

We are now ready to define Scalismo's registration object.
```scala mdoc
 val registration = Registration(metric, regularizer, regularizationWeight = 0.1, optimizer)
 
```

Registration is an iterative process. Consequently, we work with the registration using an iterator. We obtain an iterator by 
calling the `iterator` method, where we also provide a starting position for the iteration (which is in this case the zero vector):
```scala mdoc
 val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)
 val registrationIterator = registration.iterator(initialCoefficients)
```

Before running the registration, we change the iterator such that it prints in each iteration to current objective value, 
and updates the visualization. This lets us visually inspect the progress of the registration procedure.

```scala mdoc
val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
  println(s"object value in iteration $itnum is ${it.value}")
  ssmView.shapeModelTransformationView.shapeTransformationView.coefficients = it.parameters
  it
}
```

Note that the above code does not yet run the registration. It simply returns a new iterator, which augments
the original iteration with visualization. The actual registration is executed once we "consume" the iterator. 
This can, for example be achieved by converting it to a sequence. The resulting sequence holds all the intermediate 
states of the registration. We are usually only interested in the last one:

```scala mdoc
val registrationResult = visualizingRegistrationIterator.toSeq.last
```

You should see in the graphical user interface, how the face mesh slowly adapts to the shape of the target mesh.

The final mesh representation can be obtained by obtaining the transform corresponding to the parameters and to 
warp the reference mesh with this tranform:
```scala mdoc
    val registrationTransformation = transformationSpace.transformForParameters(registrationResult.parameters)
    val fittedMesh = referenceMesh.transform(registrationTransformation)
```

### Working with the registration result

The fittedMesh that we obtained above is a surface that approximates the target surface.  It corresponds to the best representation of the target in the model. For most tasks, this approximation is sufficient.
However, sometimes, we need an exact representation of the target mesh. This can be achieved by defining a projection function, which projects each point onto its closest point on the target.

```scala mdoc
val targetMeshOperations = targetMesh.operations
val projection = (pt : Point[_3D]) => {
  targetMeshOperations.closestPointOnSurface(pt).point
}
```
Composing the result of the registration with this projection, will give us a mapping that identifies for each point of the reference mesh the corresponding point of the target mesh.

```scala mdoc
val finalTransformation = registrationTransformation.andThen(projection)
```

To check this last point, we warp the reference mesh with the finalTransform and visualize it. Note that the projected target now coincides with the target mesh..

```scala mdoc
val projectedMesh = referenceMesh.transform(finalTransformation)
val resultGroup = ui.createGroup("result")
val projectionView = ui.show(resultGroup, projectedMesh, "projection")
```

### Improving registrations for more complex shapes.

This registration procedure outlined above works reasonably well for simple cases. In complex cases, in particular if you have large
shape variations, you may find it difficult to find a suitable regularization weight. When you choose the regularization weight
large, the procedure will result in a nice and smooth mesh, but fails to closely fit the surface. If you choose it small, it may
result in folds and bad correspondences. In such cases it has proven extremely useful to simply iterate the registration procedure,
with decreasing regularization weights. In the following we illustrate this procedure. We start by defining a case class, which
collects all relevant parameters:
```scala mdoc
 case class RegistrationParameters(regularizationWeight : Double, numberOfIterations : Int, numberOfSampledPoints : Int)
```
We put all the registration code into a function, which takes (among others) the registration parameters as an argument.
```scala mdoc

    def doRegistration(
            lowRankGP : LowRankGaussianProcess[_3D, Vector[_3D]],
            referenceMesh : TriangleMesh[_3D],
            targetmesh : TriangleMesh[_3D],
            registrationParameters : RegistrationParameters,
            initialCoefficients : DenseVector[Double]
        ) : DenseVector[Double] =
    {
        val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
        val fixedImage = referenceMesh.operations.toDistanceImage
        val movingImage = targetMesh.operations.toDistanceImage
        val sampler = UniformMeshSampler3D(
            referenceMesh,
            registrationParameters.numberOfSampledPoints
            )
        val metric = MeanSquaresMetric(
            fixedImage,
            movingImage,
            transformationSpace,
            sampler
            )
        val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
        val regularizer = L2Regularizer(transformationSpace)
        val registration = Registration(
            metric,
            regularizer,
            registrationParameters.regularizationWeight,
            optimizer
            )
        val registrationIterator = registration.iterator(initialCoefficients)
        val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
              println(s"object value in iteration $itnum is ${it.value}")
              ssmView.shapeModelTransformationView.shapeTransformationView.coefficients = it.parameters
              it
        }
        val registrationResult = visualizingRegistrationIterator.toSeq.last
        registrationResult.parameters
    }
```
Finally, we define the parameters and run the registration. Note that when we decrease the
regularization weight, we typically need to sample less points from the surface.

```scala mdoc
    val registrationParameters = Seq(
        RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 20, numberOfSampledPoints = 1000),
        RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 50, numberOfSampledPoints = 1000),
        RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 100, numberOfSampledPoints = 2000),
        RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 100, numberOfSampledPoints = 4000)
    )

    val finalCoefficients = registrationParameters.foldLeft(initialCoefficients)((modelCoefficients, regParameters) =>
            doRegistration(lowRankGP, referenceMesh, targetMesh, regParameters, modelCoefficients))
```
From this point we use the procedure described above to work with the registration result.

```scala mdoc
  ui.close()
```