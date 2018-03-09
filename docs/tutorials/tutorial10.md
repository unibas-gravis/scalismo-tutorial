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

# Iterative Closest Points for rigid alignment

The goal in this tutorial is to experiment with a first application of the Iterative Closest Points (ICP) algorithm in the context of rigid alignment of shapes.

In case you would like to learn more about the syntax of the Scala programming language, have a look at the **Scala Cheat Sheet**. This document is accessible under: 

*Documents -> Scala Cheat Sheet*

###### Note: for optimal performance and to keep the memory footprint low, we recommend to restart Scalismo Lab at the beginning of every tutorial.


## Automatic rigid alignment

Let us now revisit our example case of misaligned meshes: 

```tut:silent
val paola = MeshIO.readMesh(new File("datasets/Paola.stl")).get
show(paola, "Paola")

val target = MeshIO.readMesh(new File("datasets/323.stl")).get
show(target, "target")
```

As you can see here, both loaded meshes are misaligned. If we now want to align Paola's mesh to the target, we could of course manually click some corresponding landmarks and perform Procrustes analysis as we saw in a previous tutorial. 

The goal in this tutorial is however different. Here, we will try to use the Iterative Closest Point (ICP) method to perform this rigid alignment step **automatically**.

### Candidate correspondences 

Finding the best rigid transformation when given correct correspondences can be done in a closed-form solution using Procrustes analysis.

The first main idea behind the ICP algorithm is that, even though we do not have correct correspondences, we can nevertheless propose **candidate** correspondences and use the closed form solution to find a candidate rigid transformation.


To do so, let us start by selecting a set of points from Paola's mesh:

```tut:silent
val ptIds = (0 until paola.numberOfPoints by 50).map(i => PointId(i))
show(ptIds.map(id => paola.point(id)), "selected")
```

Here, we selected one in every fifty point identifiers on Paola's mesh to obtain a uniformly distributed sample of points on the surface.

**How can we find candidate correspondences for this selected set of points?**

A very basic method, in this example case, is to attribute to each selected point its closest point on the target mesh as a candidate corresponding point.


Below, we define a function that does exactly that:

```tut:silent
def attributeCorrespondences(toTransform: TriangleMesh) : IndexedSeq[(Point[_3D], Point[_3D])] = {
  ptIds.map{ id : PointId => 
    val pt = toTransform.point(id)
    val candidate = target.findClosestPoint(pt).point
    (pt, candidate)
  } 
}
```

Given a triangle mesh to be rigidly aligned to the target (in our case this will be Paola's mesh), this function loops on the point identifiers selected above and associates to each selected point on the mesh to be transformed, its closest point on the target.

The function therefore returns an indexed sequence of point tuples, where the elements of the tuple are candidate corresponding points on the mesh to be transformed and on the target respectively.


Let us now visualize the the candidate correspondences to our selected set of points on Paola's mesh:


```tut:silent
val candidateCorr = attributeCorrespondences(paola)
val targetPoints = candidateCorr.map(tuple => tuple._2)
show(targetPoints, "candidateCorrespondences")
```

As you can see in the 3D scene, the obtained candidate correspondences are clearly not good correspondences as they tend to focus on only one side of the target face. 

Let us nevertheless apply Procrustes analysis based on these candidate correspondences and retrieve a candidate rigid transformation to align Paola:


```tut:silent
val rigidTrans =  LandmarkRegistration.rigid3DLandmarkRegistration(candidateCorr)
val transformed = paola.transform(rigidTrans) 
show(transformed, "aligned?")
```


**Well, no surprise here.** Given the poor quality of the candidate correspondences, we obtained a poor rigid alignment. This said, when considering where we started from, that is the original position of Paola's mesh, we did get closer to the target.

This is where the second important idea of the ICP algorithm comes into play: **iteration**. 

Now that we have a new starting position that is closer to the target mesh, if we were to reiterate the procedure above, we would hope to get better candidate correspondences thus resulting in a better rigid alignment. Let's try it out: 

```tut:silent
val newCandidateCorr = attributeCorrespondences(transformed)
show(newCandidateCorr.map(tuple => tuple._2), "newCandidateCorr")
val newRigidTrans =  LandmarkRegistration.rigid3DLandmarkRegistration(newCandidateCorr)
val newTransformed = transformed.transform(newRigidTrans) 
show(newTransformed, "aligned??")
```

Here, we just reiterated all the steps performed above while simply using the transformed mesh (named *aligned?* in the 3D scene) as a new starting point for attributing candidate correspondences.

As you can see, the candidate correspondences are still clearly wrong, but start to be more spread around the target face. Also the resulting rigid transformation seems to bring our mesh a bit closer to the target. 

Let's now adapt our implementation such that we can perform an arbitrary number of iterations:


```tut:silent
def ICPRigidAlign(movingMesh: TriangleMesh, nbIter : Int) : TriangleMesh = {
  if(nbIter == 0) movingMesh
  else {
    val correspondences = attributeCorrespondences(movingMesh)
    val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences)
    val transformed = movingMesh.transform(transform) 
   
    if(nbIter % 15 == 0) {
      remove("icp_iter_"+ (135-nbIter))
      show(transformed, "icp_iter_"+ (150-nbIter))
    }

    ICPRigidAlign(transformed, nbIter-1)
  }
}
```

Here we simply wrapped the steps we performed above in a recursive function such that we can do them repeatedly.

At every recursive call of this function, we start by attributing candidate correspondences based on the latest position of Paola's transformed mesh. 

We then solve for the the best rigid transform, use it to transform the moving mesh and then recurse (transformed mesh will then be used for attributing candidate correspondences, etc ..).

As a halting condition for the recursion, we indicate a number of iterations where we stop recurring when the *nbIter* counter is at 0.  

Additionally, at every 15th iteration, the current moving mesh will be displayed in order to follow the progress of the iteration.


Let's now run it with 150 iterations: 

```tut:silent
//cleanup first
remove("aligned?"); remove("aligned??")
remove("candidateCorrespondences"); remove("newCandidateCorr")

val rigidfit = ICPRigidAlign(paola, 150)
show(rigidfit, "ICP_rigid_fit")
remove("icp_iter_135")
```

As you can see here, the quality of the candidate correspondences did indeed increase at every iteration thus resulting in a proper **automatic** rigid alignment of Paola to the target.

One should not forget, however, that the ICP method is very prone to local minima.

##### Exercise: all is not bright under the sun! Try changing the number of selected points for attributing candidate correspondences (i.e. compute a new *ptIds*, by taking every 100th point for example), redefine the *attributeCorrespondences* method to take the new ids into account, run the rigid ICP again and see what happens. 


###### Solution: hopefully you see that by changing the number of selected points for the ICP (e.g. increasing it from 50 to 100), the rigid ICP quickly converges to a local minimum out of which it does not exit:


```tut:silent
val ptIds = (0 until paola.numberOfPoints by 100).map(i => PointId(i))

def attributeCorrespondences(toTransform: TriangleMesh) : IndexedSeq[(Point[_3D], Point[_3D])] = {
  ptIds.map{ id : PointId => 
    val pt = toTransform.point(id)
    val candidate = target.findClosestPoint(pt).point
    (pt, candidate)
  } 
}
def ICPRigidAlign(movingMesh: TriangleMesh, nbIter : Int) : TriangleMesh = {
  if(nbIter == 0) movingMesh
  else {
    val correspondences = attributeCorrespondences(movingMesh)
    val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences)
    val transformed = movingMesh.transform(transform) 

    if(nbIter % 15 == 0) {
      remove("icp_iter_"+ (135-nbIter))
      show(transformed, "icp_iter_"+ (150-nbIter))
    }

    ICPRigidAlign(transformed, nbIter-1)
  }
}
//cleanup first

val rigidfit = ICPRigidAlign(paola, 150)
show(rigidfit, "ICP_rigid_fit")
remove("icp_iter_135")
```



<br><br>

```tut:invisible
gui.close
```