{% include head.html %}

# Model fitting using MCMC - The basic framework

In this tutorial we show how Bayesian model fitting using Markov Chain Monte Carlo can be done in Scalismo. To be able
to focus on the main components of the framework instead of technical details, we start in this tutorial with a simple toy example, which has nothing
to do with shape modelling. The application to shape modelling is discussed in depth in the next tutorial.

##### Preparation

As in the previous tutorials, we start by importing some commonly used objects and initializing the system.

```scala
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}

scalismo.initialize()
implicit val rng = scalismo.utils.Random(42)
```

### Problem setting

The problem we are considering here is a simple toy problem: We are trying to fit a (univariate) normal distribution,
with unknown mean and unknown standard deviation to a set of data points.


We test our method on synthetically generated data which is simulated from a normal distribution $$N(-5, 17)$$.

```scala
  val mu = -5
  val sigma = 17

  val trueDistribution = breeze.stats.distributions.Gaussian(mu, sigma)
  val data = for (_ <- 0 until 100) yield {
    trueDistribution.draw()
  }
```

In the following we will denote the unknown parameters by $$\theta$$; I.e. $$\theta = (\mu, \sigma)$$ and the observed data points
by $$y$$. Formally, our task is to compute the *posterior distribution*
$$p(\theta | y) = \frac{p(\theta) p(y | \theta)}{p(y)}$$ where $$p(\theta)$$ is a prior distribution over the parameters,
which we will define later.

*Remark: Computing the posterior distribution of the parameters will also be our goal in a real shape model fitting application.
The only difference is that the parameters $$\theta$$ are not mean and standarddeviation, but the shape model parameters, and the data $$y$$ are
not simulated numbers, but measurements of the target object, such as a set of landmark points, a surface or even an image.*


### Metropolis Hastings Algorithm

The way we approach such fitting problem in Scalismo is by using the
Metropolis Hastings algorithm. The Metropolis-Hastings algorithm allows us to
draw samples from any distribution, given that the unnormalized distribution can be evaluated point-wise. This requirement is
easy to fulfill for all shape modelling applications.

For setting up the Metropolis-Hastings algorithm, we need two things:
1. The (unnormalized) target distribution, from which we want to sample. In our case this is the posterior distribution $$p(\theta \mid y)$$
2. A proposal distribution $$(Q(\theta' \mid \theta))$$, which generates for a given sample $$\theta$$ a new sample $$\theta'$$.

The Metropolis Hastings algorithm introduces an ingenious scheme for accepting
and rejecting the samples from this proposal distribution, based on their probability under the target density,
such that the resulting sequence of samples is guaranteed to be distributed according to the
target distribution.

Before we discuss how we set up these components in Scalismo, we first define a class for representing
the parameters $$\theta = (\mu, \sigma)$$:

```scala
case class Parameters(mu : Double, sigma: Double)
```

We introduce a further class to represent a sample from the chain. A sample is
simply a set of parameters together with a tag, which helps us to keep track later
on, which proposal generator generated the sample:

```scala
case class Sample(generatedBy : String,
                  parameters : Parameters)
```

#### Evaluators: Modelling the target density

In Scalismo, the target density is represented by classes, which we will refer to
as *Evaluators*. Any Evaluator is a subclass of the class ```DistributionEvalutor```,
defined as follows:

```scala
trait DistributionEvaluator[A] {
  /** log probability/density of sample */
  def logValue(sample: A): Double
}
```

We see that the only thing we need to define is the log probability of a sample.

In our case, we will define separate evaluators for the prior distribution $$p(\theta)$$ and
the likelihood $$p(y | \theta)$$.
The evaluator for the likelihood is simple: Assuming a normal model, we define
the normal distribution with the given parameters $$\theta$$, and use this model
to evaluate the likelihood of the individual observations.  
We assume that the observations are i.i.d. and hence the joint probability
factorises as
$$p(y |\theta) = p(y_1, \ldots, y_n |\theta) = \prod_{i=1}^n p(y_i |\theta)$$.
This leads to the following implementation of the liklihood function:

```scala
  case class LikelihoodEvaluator(data : Seq[Double]) extends DistributionEvaluator[Sample] {

    override def logValue(theta: Sample): Double = {
      val likelihood = breeze.stats.distributions.Gaussian(
        theta.parameters.mu, theta.parameters.sigma
      )
      val likelihoods = for (x <- data) yield {
        likelihood.logPdf(x)
      }
      likelihoods.sum
    }
  }
```

Notice that we work in Scalismo with log probabilities, and hence the product in above formula
becomes a sum.

As a prior, we also use for both parameters a univariate normal distribution.

```scala
object PriorEvaluator extends DistributionEvaluator[Sample] {

    val priorDistMu = breeze.stats.distributions.Gaussian(0, 20)
    val priorDistSigma = breeze.stats.distributions.Gaussian(0, 100)

    override def logValue(theta: Sample): Double = {
      priorDistMu.logPdf(theta.parameters.mu)
      + priorDistSigma.logPdf(theta.parameters.sigma)
    }
  }
```

The target density (i.e. the posterior distribution) can be computed by
taking the product of the prior and the likelihood.

```scala
val posteriorEvaluator = ProductEvaluator(PriorEvaluator, LikelihoodEvaluator(data))
```

Note that the posteriorEvaluator represents the unnormalized posterior, as we did
not normalize by the probability of the data $$p(y)$$.

#### The proposal generator

In Scalismo, a proposal generator is defined by extending the trait
*ProposalGenerator*, which is defined as follows

```scala
trait ProposalGenerator[A] {
  /** draw a sample from this proposal distribution, may depend on current state */
  def propose(current: A): A
}
```

In order to be able to use a proposal generator in the Metropolis-Hastings algorithm,
we also need to implement the trait ```TransitionProbability```:

```scala
trait TransitionProbability[A] extends TransitionRatio[A] {
  /** rate of transition from to (log value) */
  def logTransitionProbability(from: A, to: A): Double
}
```

To keep things simple, we use here a *random walk proposal*. This is a proposal
which updates the current state by taking a step of random length in a random direction.
It is defined as follows:

```scala
case class RandomWalkProposal(stddevMu: Double, stddevSigma : Double)(implicit rng : scalismo.utils.Random)
    extends ProposalGenerator[Sample] with TransitionProbability[Sample] {

    val stepDistMu = breeze.stats.distributions.Gaussian(0, stddevMu)
    val stepDistSigma = breeze.stats.distributions.Gaussian(0, stddevSigma)

    override def propose(sample: Sample): Sample = {
      val newParameters = Parameters(
        mu = sample.parameters.mu + rng.breezeRandBasis.gaussian(0, stddevMu).draw(),
        sigma = sample.parameters.sigma + rng.breezeRandBasis.gaussian(0, stddevSigma).draw()
      )

      Sample(s"randomWalkProposal ($stddevMu, $stddevSigma)", newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) : Double = {
      val residualMu = to.parameters.mu - from.parameters.mu
      val residualSigma = to.parameters.sigma - from.parameters.sigma
      stepDistMu.logPdf(residualMu)  + stepDistMu.logPdf(residualSigma)
    }
  }
```

*Remark: the second constructor argument ```implicit rng : scalismo.utils.Random```
is used to automatically pass the globally defined random generator object to the
class. If we always use this random generator to generate our random numbers, we can obtain reproducible runs,
by seeding this random generator at the beginning of our program.*

Let's define two random walk proposals with different step length:

```scala
val smallStepProposal = RandomWalkProposal(3.0, 1.0)
val largeStepProposal = RandomWalkProposal(9.0, 3.0)
```

Varying the step length allow us to sometimes take large step, in order to explore the global
landscape, and sometimes smaller steps, to explore a local environment. We can combine these proposal into a
```MixtureProposal```, which chooses the individual proposals with a given
probability. Here We choose to take the large step 20% of the time, and the smaller
steps 80% of the time:

```scala
val generator = MixtureProposal.fromProposalsWithTransition[Sample](
    (0.8, smallStepProposal), 
    (0.2, largeStepProposal)
    )
```


#### Building the Markov Chain

Now that we have all the components set up, we can assemble the Markov Chain.

```scala
val chain = MetropolisHastings(generator, posteriorEvaluator)
```

To run the chain, we obtain an iterator,
which we then consume to drive the sampling generation. To obtain the iterator, we need to specify the initial
sample:

```scala
val initialSample = Sample(generatedBy="initial", Parameters(0.0, 10.0))
val mhIterator = chain.iterator(initialSample)
```

Our initial parameters might be far away from a high-probability area of our target
density. Therefore it might take a few hundred or even a few thousand iterations before the produced samples
start to follow the required distribution. We therefore have to drop the
samples in this burn-in phase, before we use the samples:

```scala
val samples = mhIterator.drop(1000).take(5000).toIndexedSeq  
```

As we have generated synthetic data, we can check if the expected value, computed
from this samples, really corresponds to the parameters from which we sampled
our data:

```scala
val estimatedMean = samples.foldLeft(0.0)((sum, sample) => sum + sample.parameters.mu) / samples.size
// estimatedMean: Double = -4.93328507579315
  println("estimated mean is " + estimatedMean)
// estimated mean is -4.93328507579315
  val estimatedSigma = samples.foldLeft(0.0)((sum, sample) => sum + sample.parameters.sigma) / samples.size
// estimatedSigma: Double = 16.34666970915081
  println("estimated sigma is " + estimatedSigma)
// estimated sigma is 16.34666970915081
```

In the next tutorial, we see an example of how the exact same  mechanism can be used for
fitting shape models. Before we discuss this, we should, however, spend some time
to discuss how the chain can be debugged in case something goes wrong.
You can safely skip this section and come back to it later if you first want to
see a practical example.

#### Debugging the markov Chain


Sometimes a chain does not work as expected. The reason is usually that our proposals
are not suitable for the target distribution. To diagnose the
behaviour of the chain we can introduce a logger. To write a logger, we need to extend
the trait ```AcceptRejectLogger```, which is defined as follows:

```scala
trait AcceptRejectLogger[A] {
  def accept(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit

  def reject(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit
}
```

The two methods, ```accept``` and ```reject``` are called whenever a sample is
accepted or rejected. We can overwrite these methods to implement our debugging code.


The following, very simple logger counts all the accepted and rejected samples and
computes the acceptance ratio. This acceptance ratio is a simple, but already useful
indicator to diagnose if all proposal generators function as expected.

```scala
  class Logger extends AcceptRejectLogger[Sample] {
    private val numAccepted = collection.mutable.Map[String, Int]()
    private val numRejected = collection.mutable.Map[String, Int]()

    override def accept(current: Sample,
                        sample: Sample,
                        generator: ProposalGenerator[Sample],
                        evaluator: DistributionEvaluator[Sample]
                       ): Unit = {
      val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
      numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
    }

    override def reject(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]
                         ): Unit = {
      val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
      numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
    }


    def acceptanceRatios() : Map[String, Double] = {
      val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
      val acceptanceRatios = for (generatorName <- generatorNames ) yield {
        val total = (numAccepted.getOrElse(generatorName, 0)
                     + numRejected.getOrElse(generatorName, 0)).toDouble
        (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
      }
      acceptanceRatios.toMap
    }
  }

```

To use the logger, we simply rerun the chain, but pass the logger now as
a second argument to the ```iterator``` method:

```scala
val logger = new Logger()
val mhIteratorWithLogging = chain.iterator(initialSample, logger)
 val samples2 = mhIteratorWithLogging.drop(1000).take(3000).toIndexedSeq  
```

We can now check how often the individual samples got accepted.

```scala
println("acceptance ratio is " +logger.acceptanceRatios())
// acceptance ratio is Map(randomWalkProposal (3.0, 1.0) -> 0.44609079445145017, randomWalkProposal (9.0, 3.0) -> 0.11729141475211609)
```

We see that the acceptance ratio of the random walk proposal, which takes the
smaller step is quite high, but that the larger step is often rejected. We might
therefore want to reduce this step size slightly, as a proposal that is so often
rejected is not very efficient.

In more complicated applications, this type of debugging is crucial for obtaining
efficient fitting algorithms.
