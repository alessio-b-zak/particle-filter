#include <Rcpp.h>
using namespace Rcpp;


void nonResampleParticles(Rcpp::NumericMatrix& rParticles,
                       Rcpp::NumericMatrix& particles,
                       Rcpp::NumericVector& weights){
  Rcpp::IntegerVector indices = Rcpp::sample(particles.ncol(),particles.ncol(), true, weights, false);
  for(int i = 0; i < particles.ncol() ; i++) {
    rParticles(_, i) = particles(_,indices(i));
  }
}

void nonInitialiseVariables(Rcpp::NumericMatrix& resampledParticles, Rcpp::NumericVector& initialState){
  for(int col = 0 ; col < resampledParticles.ncol(); col++) {
      resampledParticles(_,col) = initialState;
  }
}

void nonSimulateTransition(Rcpp::NumericMatrix::Column particle,
                        Rcpp::NumericMatrix::Column rParticle,
                        Rcpp::NumericVector& params)
  {
  int ntrans = (int) 1 / params[1];
  int S = rParticle[0];
  int I = rParticle[1];
  int R = rParticle[2];

  for(int i = 0 ; i < ntrans ; i++) {
    double p_SI = 1 - exp(-(params[2] * I * params[1])/params[0]);
    double p_IR = 1 - exp(-params[4] * params[1]);

    int dN_SI = R::rbinom(S, p_SI);
    int dN_IR = R::rbinom(I, p_IR);

    S -= dN_SI;
    I += (dN_SI - dN_IR);
    R += dN_IR;
  }

  particle[0] = S;
  particle[1] = I;
  particle[2] = R;
}

void nonPropagateParticles(Rcpp::NumericMatrix& particles,
                        Rcpp::NumericMatrix& resampledParticles,
                        Rcpp::NumericVector& params){
  //loop through each particle and run single timestep obs
  for(int i = 0; i < particles.ncol(); i++) {
      nonSimulateTransition(particles(_,i), resampledParticles(_,i), params);
  }
}

void nonNormaliseWeights(Rcpp::NumericVector& tWeights,
                       Rcpp::NumericVector& nWeights) {
  nWeights = exp(tWeights) / sum(exp(tWeights));
}

void nonWeightParticles(double y,
                     Rcpp::NumericVector& weights,
                     Rcpp::NumericMatrix& particles,
                     Rcpp::NumericVector& params){
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

double nonComputeLikelihood(Rcpp::NumericVector& tWeights, double lWeightsMax, int n_particles){
  double tLikelihood = lWeightsMax + log(sum(exp(tWeights))) - log(n_particles);
  return tLikelihood;

}

// Particle filter for sir model
// Param vectors take form [N, dt, beta, rho, gamma]
// State vectors take  form [S, I , R]
// [[Rcpp::export(name=nonParticleFilterCpp)]]
double nonParticleFilter(Rcpp::NumericVector y,
                    Rcpp::NumericVector params,
                    int n_particles,
                    Rcpp::NumericVector initialState
                    ) {

  // Initialising
  // Marshalling
  double logLikelihood = 0;
  int n_obs = y.size();

  double lWeightsMax;
  int num_transitions = 1 / params[1];

  // Create Procedure Variables
  Rcpp::NumericMatrix particles(3, n_particles);
  Rcpp::NumericMatrix resampledParticles(3, n_particles);
  Rcpp::NumericVector weights(n_particles);
  Rcpp::NumericVector tWeights(n_particles);
  Rcpp::NumericVector nWeights(n_particles);

  nonInitialiseVariables(resampledParticles, initialState);

  for(int t = 0; t < n_obs; t++) {
    nonPropagateParticles(particles, resampledParticles, params);
    nonWeightParticles(y[t], weights, particles, params);

    lWeightsMax = Rcpp::max(weights);
    tWeights = weights - lWeightsMax;

    nonNormaliseWeights(tWeights, nWeights);

    nonResampleParticles(resampledParticles, particles, nWeights);
    logLikelihood += nonComputeLikelihood(tWeights, lWeightsMax, n_particles);
  }

  return logLikelihood;
}