#include <Rcpp.h>
#include <RcppParallel.h>
#include <sitmo.h>
#include <boost/random/binomial_distribution.hpp>
using namespace Rcpp;
using namespace RcppParallel;
using binomial = boost::random::binomial_distribution<int>;


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(sitmo)]]

void resampleParticles(Rcpp::NumericMatrix& rParticles,
                       Rcpp::NumericMatrix& particles,
                       Rcpp::NumericVector& weights,
                       int n_particles){
  // Rcout << "num particles: " << particles.ncol() << "\n";
  // int zcount = 0;
  // for(int i = 0; i < particles.ncol() ; i++) {
  //   if(weights[i] == 0) {
  //     zcount++;
  //   }
  // }
  // Rcout << "num zero weights: " << zcount << "\n";
  Rcpp::IntegerVector indices = Rcpp::sample(n_particles, n_particles, true, weights, false);
  for(int i = 0; i < n_particles ; i++) {
    rParticles(_, i) = particles(_,indices(i));
  }
}

void initialiseVariables(Rcpp::NumericMatrix& resampledParticles, Rcpp::NumericVector& initialState){
  for(int col = 0 ; col < resampledParticles.ncol(); col++) {
      resampledParticles(_,col) = initialState;
  }
}

void simulateTransition(RcppParallel::RMatrix<double>::Column particle, RcppParallel::RMatrix<double>::Column rParticle,
                        RcppParallel::RVector<double> params,
                        sitmo::prng_engine& engine,
                        binomial& transitions)
  {
  int ntrans = (int) 1 / params[1];
  int S = rParticle[0];
  int I = rParticle[1];
  int R = rParticle[2];

  for(int i = 0 ; i < ntrans ; i++) {
    double p_SI = 1 - exp(-(params[2] * I * params[1])/params[0]);
    double p_IR = 1 - exp(-params[4] * params[1]);

    int dN_SI = transitions(engine, binomial::param_type(S, p_SI));
    int dN_IR = transitions(engine, binomial::param_type(I, p_IR));

    S -= dN_SI;
    I += (dN_SI - dN_IR);
    R += dN_IR;
  }


  particle[0] = S;
  particle[1] = I;
  particle[2] = R;
}

void normaliseWeights(Rcpp::NumericVector& tWeights,
                       Rcpp::NumericVector& nWeights) {
  nWeights = exp(tWeights) / sum(exp(tWeights));
}

void propagateParticles(RcppParallel::RMatrix<double>& particles,
                        RcppParallel::RMatrix<double>& resampledParticles,
                        RcppParallel::RVector<double>& params,
                        sitmo::prng_engine& engine,
                        binomial transitions,
                        int n_particles){ //loop through each particle and run single timestep obs
  for(int i = 0; i < n_particles; i++) {
      simulateTransition(particles.column(i), resampledParticles.column(i), params, engine, transitions);
  }
}

void weightParticles(double y,
                     Rcpp::NumericVector& weights,
                     Rcpp::NumericMatrix& particles,
                     Rcpp::NumericVector& params
                     ){
  for(int i = 0; i < weights.size(); i++) {
    Rcpp::NumericMatrix::Column states = particles(_,i);
    weights[i] = R::dpois(y, params[3] * states[1], 1);

    //  if(weights[i]==0) {
    //  Rcout << "----" << "\n";
    //  Rcout << "rho is: " << params[3] << "\n";
    //  Rcout << "I is: " << states[1] << "\n";
    //  Rcout << "y is: " << y << "\n"; 
    //  Rcout << "----" << "\n";
    //  }
  }
}

double computeLikelihood(Rcpp::NumericVector& tWeights, double lWeightsMax, int n_particles){
  double tLikelihood = lWeightsMax + log(sum(exp(tWeights))) - log(n_particles);
  return tLikelihood;
}

// Particle filter for sir model
// Param vectors take form [N, dt, beta, rho, gammalogiiikjkjik]
// State vectors take  form [S, I , R]
// [[Rcpp::export(name=particleFilterCpp)]]
double particleFilter(Rcpp::NumericVector y,
                    Rcpp::NumericVector params,
                    int n_particles,
                    Rcpp::NumericVector initialState
                    ) {

  // Initialising
  // Marshalling
  double logLikelihood = 0;
  double otherLikelihood = 0;
  int n_obs = y.size();

  int num_transitions = 1 / params[1];

  // Create Procedure Variables
  Rcpp::NumericMatrix particles(3, n_particles);
  RcppParallel::RMatrix<double> oParticles(particles);

  Rcpp::NumericMatrix resampledParticles(3, n_particles);
  RcppParallel::RMatrix<double> oRParticles(resampledParticles);

  Rcpp::NumericVector weights(n_particles);
  RcppParallel::RVector<double> oWeights(weights);

  Rcpp::NumericVector nWeights(n_particles);
  RcppParallel::RVector<double> oNWeights(weights);
  
  Rcpp::NumericVector tWeights(n_particles);
  RcppParallel::RVector<double> oTWeights(weights);

  RcppParallel::RVector<double> oParams(params);

  double lWeightsMax;

  uint32_t coreSeed = static_cast<uint32_t>(runif(1,1.0, std::numeric_limits<uint32_t>::max())[0]);
  sitmo::prng_engine engine(coreSeed);

  binomial transitions;

  initialiseVariables(resampledParticles, initialState);

  for(int t = 0; t < n_obs; t++) {
    propagateParticles(oParticles, oRParticles, oParams, engine, transitions, n_particles);

    weightParticles(y[t], weights, particles, params );
    lWeightsMax = Rcpp::max(weights);
    tWeights = weights - lWeightsMax;

    normaliseWeights(tWeights, nWeights);
    resampleParticles(resampledParticles, particles, nWeights, n_particles);
    double ll = computeLikelihood(tWeights, lWeightsMax, n_particles);
    logLikelihood += ll;
  }


  return logLikelihood;
}
