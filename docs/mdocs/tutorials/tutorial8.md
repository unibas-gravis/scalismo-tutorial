
# Posterior Shape Models

In this tutorial we will use Gaussian processes for regression tasks and experiment with the concept of posterior shape models.

As always, we start with importing the commonly used objects and classes and initializing the system.

```scala mdoc
import scalismo.geometry.{_, Vector=>SpatialVector}
import scalismo.common._
import scalismo.ui.api._
import breeze.linalg.{DenseMatrix, DenseVector}

scalismo.initialize()

implicit val rng = scalismo.utils.Random(42)

val ui = ScalismoUI()
```

We also load and visualize the face model:
```scala mdoc
val model = StatisticalModelIO.readStatistialMeshModel(new File("datasets/bfm.h5")).get

val group = ui.createGroup("modelGroup")
val ssmView = ui.show(modelGroup, model, "model")
```


## Fitting observed data using Gaussian process regression

The reason we build statistical models, is that we want to use them 
for explaining data. More precisely, given some observed data, we fit the model
to the data and get as a result a distribution over the model parameters. 
In our case, the model is a Gaussian process model of shape deformations, and the data are observed shape deformations; I.e. deformation vectors from the reference surface. 

To illustrate this process, we simulate some data. We generate  
a deformation vector at the tip of the nose, which corresponds ot a really long
nose:

```scala mdoc
val tipNoseID = PointId(8156)
val noseTipReference = model.referenceMesh.pointsSet.point(tipNoseId)
val noseTipMean = model.mean.pointsSet.point(tipNoseId)
val noseTipDeformation = (noseTipMean - noseTipReference) * 2.0

val noseTipDeformationField = DiscreteField3D(UnstructuredPointsDomain3D(noseTipReference), noseTipDeformation)

val observationGroup = ui.createGroup("observation")
ui.show(observationGroup, noseTipDeformationField, "noseTip")
```

The Gaussian process model assumes, that the deformation is always only observed with some uncertainty, which can be modelled using a normal distribution. 
```scala mdoc
val noise = NDimensionalNormalDistribution(Vector3D(0,0,0), SquareMatrix3D.eye)
```
We can now obtain the regression result by feeding this data to the method ```regression``` of the ```GaussianProcess``` object:

```scala mdoc
val gp = model.gp.interpolate(NearestNeighborInterpolator3D())
val posteriorGP : LowrankGaussianProcess[_3D, UnstructuredPointsDaomain, SpatialVector[_3D]] = GaussianProcess.regression(gp, noseTipDeformationField, noise)
```

Note that the result of the regression is again a Gaussian process, over the same domain as the original process. We call this the *posterior process*. 
This construction is very important in Scalismo. Therefore, we have a convenience method defined directly on the Gaussian process object. We could write the same in 
the more succinctly:
```scala mdoc
gp.posterior(noseTipDeformationField, noise)
```
```

Independendently of how you call the method, the returned type is a continuous Gaussian Process from which we can now sample deformations at any set of points:

```scala mdoc
val posteriorSample : DiscreteVectorField[_3D, SpatialVector[_3D]] = posterior.sampleAtPoints(model.referenceMesh)
val posteriorSampleGroup = ui.createGroup("posteriorSamples")
for (i <- 0 until 10) {
    ui.show(posteriorSampleGroup, posteriorSample, "posteriorSample")
}
```


### Posterior of a StatisticalMeshModel:

Given that the StatisticalMeshModel is merely a wrapper around a GP, the same posterior functionality is available for statistical mesh models:

```scala mdoc
val discreteTrainingData = IndexedSeq((PointId(8156), furthestPoint, littleNoise))
val discretePosterior : StatisticalMeshModel = model.posterior(discreteTrainingData)
```

Notice in this case, since we are working with a discrete Gaussian process, the observed data is specified in terms of the *point identifier* of the nose tip point instead of its 3D coordinates. 

Let's visualize the obtained posterior model:

```tut:silent
val posteriorModelGroup = ui.group("posteriorModel")
show(posteriorModelGroup, posteriorModelGroup, "NoseyModel")
```

##### Exercise: sample a few random faces from the graphical interface using the random button. Notice how all faces display large noses :) with the tip of the nose remaining close to the selected landmark.


Here again we obtain much more than just a single face instance fitting the input data: we get a full normal distribution of shapes fitting the observation. The **most probable** shape, and hence our best fit, is the **mean** of the posterior.

We notice by sampling from the posterior model that we tend to get faces with rather large noses. This is since we chose our observation to be twice the length of the 
average (mean) deformation at the tip of the nose.


#### Landmark uncertainty:

As you could see previously, when specifying the training data for the posterior GP computation, we model the uncertainty of the input data.
This can allow to tune the fitting results.

Below, we perform the posterior computation again with, this time, a 5 times bigger noise variance.


```tut:silent
val largenoise = NDimensionalNormalDistribution(Vector3D(0,0,0), SquareMatrix.eye * 5.0))
val discreteTrainingDataN = IndexedSeq((PointId(8156), furthestPoint, largenoise))
val discretePosteriorN = model.posterior(discreteTrainingDataN)
show(discretePosteriorN, "NoisyNoseyModel")
```
##### Exercise: sample a few faces from this noisier posterior model. How flexible is the model compared to the previous posterior? How well are the sample tips fitting to the indicated landmark when compared with the previous posterior?


```scala mdoc
ui.close
```
