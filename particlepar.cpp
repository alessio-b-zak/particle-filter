// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <RcppParallel.h>
#include <dqrng_distribution.h>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <xoshiro.h>
#include <omp.h>
using namespace arma;
using namespace Rcpp;
using binomial = boost::random::binomial_distribution<int>;
using poisson = boost::random::poisson_distribution<int>;


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(dqrng)]]
// [[Rcpp::depends(sitmo)]]

void resampleParticles(Rcpp::NumericMatrix& rParticles,
                       Rcpp::NumericMatrix& particles,
                       Rcpp::NumericVector& weights){
  // Rcout << "num particles: " << particles.ncol() << "\n";
  // int zcount = 0;
  // for(int i = 0; i < particles.ncol() ; i++) {
  //   if(weights[i] == 0) {
  //     zcount++;
  //   }
  // }
  // Rcout << "num zero weights: " << zcount << "\n";
  Rcpp::IntegerVector indices = Rcpp::sample(particles.ncol(),particles.ncol(), true, weights, false);
  for(int i = 0; i < particles.ncol() ; i++) {
    rParticles(_, i) = particles(_,indices(i));
  }
}

void initialiseVariables(Rcpp::NumericMatrix& resampledParticles, Rcpp::NumericVector& initialState){
  for(int col = 0 ; col < resampledParticles.ncol(); col++) {
      resampledParticles(_,col) = initialState;
  }
}

void simulateTransition(RcppParallel::RMatrix<double>::Column particle,
                        RcppParallel::RMatrix<double>::Column rParticle,
                        RcppParallel::RVector<double> params,
                        dqrng::xoshiro256plus& lrng)
  {
  int ntrans = (int) 1 / params[1];
  int S = rParticle[0];
  int I = rParticle[1];
  int R = rParticle[2];

//   Rcout << "S " << S  << "\n";
//   Rcout << "I " << I  << "\n";
//   Rcout << "R " << R  << "\n";
  for(int i = 0 ; i < ntrans ; i++) {
    double p_SI = 1 - exp(-(params[2] * I * params[1])/params[0]);
    double p_IR = 1 - exp(-params[4] * params[1]);
    
    boost::random::binomial_distribution<int> distSI(S, p_SI);
    auto genSI = std::bind(distSI, std::ref(lrng));

    boost::random::binomial_distribution<int> distIR(I, p_IR);
    auto genIR = std::bind(distIR, std::ref(lrng));

    // binomial d;

    int dN_SI = genSI();
    int dN_IR = genIR();

    // d(lrng, binomial::param_type(S, p_SI));
    // d(lrng, binomial::param_type(I, p_IR));

    // Rcout << "iteration " << i << "dN_IR = "  << dN_IR << "\n";
    // Rcout << "iteration " << i << "dN_SI = "  << dN_SI << "\n";

    S -= dN_SI;
    I += (dN_SI - dN_IR);
    R += dN_IR;
  }

  particle[0] = S;
  particle[1] = I;
  particle[2] = R;
}

void propagateParticles(RcppParallel::RMatrix<double> particles,
                        RcppParallel::RMatrix<double> resampledParticles,
                        RcppParallel::RVector<double> params,
                        dqrng::xoshiro256plus& rng){
  //loop through each particle and run single timestep obs
   #pragma omp parallel 
   {
    dqrng::xoshiro256plus lrng(rng);
    lrng.jump(omp_get_thread_num() + 1);

   #pragma omp for
    for(int i = 0; i < particles.ncol(); i++) {
        simulateTransition(particles.column(i), resampledParticles.column(i), params, rng);
    }
   }
}

void weightParticles(double y,
                     Rcpp::NumericVector& weights,
                     Rcpp::NumericMatrix& particles,
                     Rcpp::NumericVector& params,
                     dqrng::xoshiro256plus& rng){
  for(int i = 0; i < weights.size(); i++) {
    Rcpp::NumericMatrix::Column states = particles(_,i);
    weights[i] = R::dpois(y, params[3] * states[1], 0);
    //  if(weights[i]==0) {
    // weights[i] = R::dpois(y, params[3] * states[1], 0);
    // Rcout << "rho is: " << params[3] << "\n";
    // Rcout << "I is: " << states[1] << "\n";
    // Rcout << "y is: " << y << "\n"; 
    // Rcout << "----" << "\n";
//   }
  }
}

double computeLikelihood(Rcpp::NumericVector& weights){
  int N = weights.size();
  return log(sum(weights)/N);
}

// Particle filter for sir model
// Param vectors take form [N, dt, beta, rho, gamma]
// State vectors take  form [S, I , R]
// [[Rcpp::export(name=particleFilterCppPar)]]
double particleFilter(Rcpp::NumericVector y,
                    Rcpp::NumericVector params,
                    int n_particles,
                    Rcpp::NumericVector initialState
                    ) {

  // Initialising
  // Marshalling
  double logLikelihood = 0;
  int n_obs = y.size();

  int num_transitions = 1 / params[1];

  // Create Procedure Variables
  Rcpp::NumericMatrix particles(3, n_particles);
  RcppParallel::RMatrix<double> oParticles(particles);
  Rcpp::NumericMatrix resampledParticles(3, n_particles);
  RcppParallel::RMatrix<double> oRParticles(resampledParticles);
  Rcpp::NumericVector weights(n_particles);
  RcppParallel::RVector<double> oWeights(weights);
  RcppParallel::RVector<double> oParams(params);

  dqrng::xoshiro256plus rng(42);

  initialiseVariables(resampledParticles, initialState);
//   propagateParticles(oParticles, oRParticles, oParams, rng);
//   weightParticles(y[0], weights, particles, params);
//   resampleParticles(resampledParticles, particles, weights);
//   logLikelihood += computeLikelihood(weights);

 for(int t = 0; t < n_obs; t++) {
   propagateParticles(oParticles, oRParticles, oParams, rng);
   weightParticles(y[t], weights, particles, params, rng);
   resampleParticles(resampledParticles, particles, weights);
   logLikelihood += computeLikelihood(weights);
 }

  return logLikelihood;
}
