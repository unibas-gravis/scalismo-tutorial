{% include head.html %}

# Bayesian Model Fitting - The basic framework

In this tutorial we show how Bayesian model fitting using Markov Chain Monte Carlo can be done in Scalismo. To be able
to focus on the main components of the framework, we start in this tutorial with a simple toy example, which has nothing
to do with shape modelling. The application to shape modelling is discussed in depth in the next tutorial. 

##### Preparation

As in the previous tutorials, we start by importing some commonly used objects and initializing the system. 

```scala mdoc:silent
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}

scalismo.initialize()
implicit val rng = scalismo.utils.Random(42)
```

### Problem setting

The problem we are considering here is a simple toy problem: We are trying to fit a (univariate) normal distribution
to a set of data points. We generate the data, which we denote in the following by $$y$$ from a normal distribution $$x \sim N(-5, 17)$$. 

```scala mdoc:silent
  val mu = -5
  val sigma = 17

  val trueDistribution = breeze.stats.distributions.Gaussian(mu, sigma)
  val data = for (_ <- 0 until 100) yield {
    trueDistribution.draw()
  }
```
Assuming a normal model, our goal is to infer the unknown parameters $$\mu$$ and $$\sigma$$ of the normal distribution from 
this data. Denoting the unknown parameters as $\theta = (\mu, \sigma)^T$, the task is to estimate the *posterior distribution*
$$p(\theta | y) = \frac{p(\theta) p(y | \theta)}{p(y)}$$ where $$p(theta)$$ is a prior distribution over the parameters, 
which we will define later.

*Remark: In a shape model fitting application, the parameters $$\theta$$ are all the model parameters and the data $$y$$ are 
observations from the target shape, such as a set of landmark points, a surface or even an image.* 


### Metropolis Hastings Algorithm

In Scalismo, the way we approach such fitting problem is by using the 
Metropolis Hastings algorithm. The Metropolis Hastings algorithm allows us to 
draw samples from any distribution. The only requirements are, that the unnormalized
distribution can be evaluated point-wise. To make the algorithm work, we need 
 to specify a proposal distribution $$Q(\theta | \theta')$$, from which we can sample. 
 The Metropolis Hastings algorithm introduces an ingenious scheme for accepting 
 and rejecting the samples from the proposal distribution, based on the target density, 
 such that the resulting sequence of samples are distributed according to the 
 target distribution. 
 
Hence to make this algorithm work in Scalismo, we need to define:
1. The (unnormalized) target distribution, from which we want to sample. Here, this is the posterior distribution $$p(\theta | y)$$. 
2. The proposal generator

Before we discuss these components in detail, we first define a class for representing 
the parameters $$\theta = (\mu, \sigma)$$:
```scala mdoc:silent
case class Parameters(mu : Double, sigma: Double)
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
We see that we only need to define the log probability of a sample:
In our case, we will define separate evaluators for the prior distribution $$p(\theta)$$ and
  the likelihood $$p(y | \theta)$$.

The evaluator for the likelihood is simple: Assuming a normal model, we define
the normal distribution with the given parameters $$\theta$$, and use this model
to evaluate the likelihood of the individual observations.  
Assuming that the observations are i.i.d. and hence the joint probability
factorizes,
$$p(y |\theta) = p(y_1, \ldots, y_n |\theta) = \prod_{i=1}^n p(y_i |theta). $$
we arrive at the following implementation
 
```scala mdoc:silent
case class LikelihoodEvaluator(data : Seq[Double]) extends DistributionEvaluator[Parameters] {

    override def logValue(theta: Parameters): Double = {
      val likelihood = breeze.stats.distributions.Gaussian(theta.mu, theta.sigma)
      val likelihoods = for (x <- data) yield {
        likelihood.logPdf(x)
      }
      likelihoods.sum
    }
}
```
Notice that we work in Scalismo with log probabilities, and hence the product in above formula
becomes a sum.

As a prior, we also use a normal distribution. We treat both parameters
as independent. 

```scala mdoc:silent
  case class PriorEvaluator() extends DistributionEvaluator[Parameters] {

    val priorDistMu = breeze.stats.distributions.Gaussian(0, 20)
    val priorDistSigma = breeze.stats.distributions.Gaussian(0, 100)

    override def logValue(theta: Parameters): Double = {
      priorDistMu.logPdf(theta.mu) + priorDistSigma.logPdf(theta.sigma)
    }
  }
```

The target density (i.e. the posterior distribution) can be computed by 
taking the product of the prior and the likelihood. 
```scala mdoc:silent
  val priorEvaluator = PriorEvaluator()
  val likelihoodEvaluator = LikelihoodEvaluator(data)
  val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)
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

To keep things simple, we use here a *random walk proposal*, that is a proposal, 
which perturbs the current state by taking a step of random length in a random direction. 
It is defined as follows:
```scala mdoc:silent
case class RandomWalkProposal(stddevMu: Double, stddevSigma : Double)(implicit rng : scalismo.utils.Random)
    extends ProposalGenerator[Parameters] with TransitionProbability[Parameters] {

    val perturbationDistrMu = breeze.stats.distributions.Gaussian(0, stddevMu)
    val perturbationDistrSigma = breeze.stats.distributions.Gaussian(0, stddevSigma)

    override def propose(theta: Parameters): Parameters = {
      Parameters(
        mu = theta.mu + rng.breezeRandBasis.gaussian(0, stddevMu).draw(),
        sigma = theta.sigma + rng.breezeRandBasis.gaussian(0, stddevSigma).draw()
      )
    }

    override def logTransitionProbability(from: Parameters, to: Parameters) : Double = {
      val residualMu = to.mu - from.mu
      val residualSigma = to.sigma - from.sigma
      perturbationDistrMu.logPdf(residualMu)  + perturbationDistrMu.logPdf(residualSigma)
    }
  }
```   

Let's define two random walk proposals with different step length:
```scala mdoc:silent
val smallStepProposal = RandomWalkProposal(0.1, 0.5)
val largeStepProposal = RandomWalkProposal(1, 5)
```
Varying the step length allow us to sometimes take larger step, to explore the global
landscape, and sometimes explore locally. We can combine these proposal into a 
```MixtureProposal```, which chooses the individual proposals with a given 
probability. Here We choose to take the large step 20% of the time, and the smaller
steps 80% of the time:

```scala mdoc:silent
val generator = MixtureProposal.fromProposalsWithTransition[Parameters](
    (0.8, smallStepProposal), 
    (0.2, largeStepProposal)
    )
```
    

#### Building the Markov Chain

Now that we have all the components set up, we can assemble the Markov Chain.
```scala mdoc:silent
val chain = MetropolisHastings(generator, posteriorEvaluator)
```

Before we run the chain, we set up a logger, which let's us diagnose the
behaviour of the chain. This is useful to understand how often the proposals
are accepted or rejected. This is done by extending from the trait 
```AcceptRejectLogger```

```scala mdoc:silent
  class Logger extends AcceptRejectLogger[Parameters] {
    var numAccepted = 0
    var numRejected = 0

    override def accept(current: Parameters, sample: Parameters, generator: ProposalGenerator[Parameters], evaluator: DistributionEvaluator[Parameters]): Unit = {
      numAccepted += 1
    }
    override def reject(current: Parameters, sample: Parameters, generator: ProposalGenerator[Parameters], evaluator: DistributionEvaluator[Parameters]): Unit = {
      numRejected += 1
    }
    def acceptanceRatio() = {
      numAccepted / (numAccepted + numRejected).toDouble
    }
  }
```
Here we simply compute the acceptance ratio. We could, however, use this logger, to 
collect much more fine grained information about the chains. 

We are finally ready to run the chain. This is done by obtaining an iterator, 
which we then consume. To obtain the iterator, we need to specify the initial 
parameters, as well as the logger:

```scala mdoc:silent
  val initialParameters = Parameters(0.0, 10.0)
  val logger = new Logger()
  val mhIterator = chain.iterator(initialParameters, logger)
```

Our initial parametes might be far away from a high-probability area of our target 
density. Therefore it might take a few (hundred) iterations before the produced samples
start to follow the required distribution. We therefore have to drop the 
samples in this burn-in phase, before we use the samples:
```scala mdoc:silent
  val thetas = mhIterator.drop(1000).take(3000).toIndexedSeq  
```
As we have generated synthetic data, we can check if the expected value, computed
from this samples, really corresponds to the parameters from which we sampled
our data: 
```scala mdoc
  val estimatedMean = thetas.foldLeft(0.0)(
    (sum, theta) => sum + theta.mu
    ) / thetas.size
  println("estimated mean is " + estimatedMean)
  val estimatedSigma = thetas.foldLeft(0.0)(
  (sum, theta) => sum + theta.sigma
  ) / thetas.size
  println("estimated sigma is " + estimatedSigma)
```
We also check the acceptance ratio is acceptable
```scala mdoc
  println("acceptance ratio is " +logger.acceptanceRatio())
```

In this case, everything is okay and we have achieved our goal.

In the next tutorial, we see an example of how we can use the mechanism for 
fitting shape models.
