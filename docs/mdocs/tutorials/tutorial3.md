# From meshes to deformation fields

*In this tutorial, we show how the deformation fields that relate two meshes can be computed and visualized.*

##### Related resources

The following resources from our [online course](https://www.futurelearn.com/courses/statistical-shape-modelling) may provide
some helpful context for this tutorial:

- Modelling Shape Deformations [(Video)](https://www.futurelearn.com/courses/statistical-shape-modelling/3/steps/250326)  


##### Preparation

As in the previous tutorials, we start by importing some commonly used objects and initializing the system. 

```scala mdoc:silent
import scalismo.geometry._
import scalismo.common._
import scalismo.ui.api._

scalismo.initialize()
implicit val rng = scalismo.utils.Random(42)

val ui = ScalismoUI()
```

We will also load three meshes and visualize them in Scalismo-ui.
```scala mdoc:silent
import scalismo.io.MeshIO

val dsGroup = ui.createGroup("datasets")

val meshFiles = new java.io.File("datasets/testFaces/").listFiles.take(3)
val (meshes, meshViews) = meshFiles.map(meshFile => {
  val mesh = MeshIO.readMesh(meshFile).get 
  val meshView = ui.show(dsGroup, mesh, "mesh")
  (mesh, meshView) // return a tuple of the mesh and the associated view
}) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews 

```

#### Representing meshes as deformations

In the following we show how we can represent a mesh as a reference mesh plus a deformation field. This is possible, 
because the meshes are all in correspondence; I.e. they all have the same number of points and points with the same id in the meshes represent
the same point/region in the mesh.

In a first step we need to treat one of the meshes, say *face_0*, as the reference mesh. 

```scala mdoc:silent
val reference = meshes(0) // face_0 is our reference
```
Now any mesh, which is in correspondence with this reference can be represented as a deformation field defined defined on this 
reference mesh (i.e. the reference mesh is its domain). 

In Scalismo, such deformation fields are represented using a ```DiscreteVectorField```, which we can create as follows. 

```scala mdoc:silent
val deformations : IndexedSeq[EuclideanVector[_3D]] = reference.pointSet.pointIds.map {
  id =>  meshes(1).pointSet.point(id) - reference.pointSet.point(id)
}.toIndexedSeq

val deformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformations)
```

Similar to discrete scalar images, a Discrete Vector Field is defined over a discrete domain. In contrast to images, the domain does not need to be structured (a grid for example) and can be any arbitrary finite set of points. In the above example code, we defined the domain to be the reference mesh points, which is of type ```UnstructuredPointsDomain[_3D]```, as we can easily check:

```scala mdoc:silent
val refDomain : UnstructuredPointsDomain[_3D] = reference.pointSet
deformationField.domain == refDomain
```

As for images, the deformation vector associated with a particular point id in a *DiscreteVectorField* can be retrieved via its point id:

```scala mdoc:silent
deformationField(PointId(0))
```

We can also directly visualize this deformation field in Scalismo-ui:

```scala mdoc:silent
val deformationFieldView = ui.show(dsGroup, deformationField, "deformations")
```
We can now verify visually that the deformation vectors point from the reference to *face_1*.
To see the effect better we need to remove *face2* from the ui, make the reference transparent

```scala mdoc:silent
meshViews(2).remove()
meshViews(0).opacity = 0.3
```

##### Exercise: generate the rest of the deformation fields that represent the rest of the faces in the dataset and display them.


### Deformation fields over continuous domains:

The deformation field that we computed above is discrete as it is defined over a finite number of mesh points. This means, the deformation 
is only defined at the mesh points. Since the real-world objects that we model are continuous, and the discretization of our meshes is rather
arbitrary, this is not ideal. In Scalismo we usually prefer to work with continuous domains. 
Whenever we have an object in Scalismo, which is defined on a discrete domain, we can obtain a continuous representation, by means
of interpolation. 

To turn our deformation field into a continuous deformation field, we need to define an ```Interpolator``` and call the ```interpolate```
method:
```scala mdoc:silent
val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
val continuousDeformationField : Field[_3D, EuclideanVector[_3D]] = deformationField.interpolate(interpolator)
```
As we do not know much about the structure of the points that define the mesh, we use a ```NearestNeighborInterpolator```, which means
that for every point on which we want to evaluate the deformation, the nearest point on the mesh is found and returned. For other type of domain, 
such as e.g. Image Domains, more sophisticated interpolation strategies are available. 

Also observe the type signature of the interpolator. We needed to provide as a type argument the type of the domain, which is in this case an ```UnstructuredPointDomain[_3D]```.

The resulting  deformation field is now defined over the entire real space and can be evaluated at any point, even if it does not belong to the reference mesh vertices.

```scala mdoc:silent
continuousDeformationField(Point(-100,-100,-100))
```

##### Exercise: Evaluate the new vector field above at the vertices of dataset(2) (*face_2*) and display the resulting vector field


##### Exercise: Compute the mesh resulting from warping every point of *face_2* with the vector field computed above and display it. Hint: you can define a transform as we did in the rigid alignment tutorial and use it to transform the mesh. (Do not expect a pretty result :))

```scala mdoc:invisible
ui.close()
```